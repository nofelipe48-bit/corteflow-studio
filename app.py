import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ---------------------------------------------------------------------------
# Configuração da aplicação e diretórios temporários
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_TEMP_DIR = Path("/tmp/corteflow_jobs")
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 3600  # Limpa cortes com mais de 60 minutos para economizar disco no Space
MAX_TRANSCRIPT_CHARS = 160_000

app = FastAPI(
    title="CorteFlow — Cortes Automáticos do YouTube",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Modelos de entrada e saída (Pydantic)
# ---------------------------------------------------------------------------
ClipStyle = Literal["viral", "educativo", "engracado"]


class CreateJobRequest(BaseModel):
    url: str = Field(..., min_length=8, description="URL completa do vídeo ou live do YouTube.")
    clip_count: int = Field(default=3, ge=3, le=5, description="Quantidade de cortes (3 a 5).")
    style: ClipStyle = Field(default="viral", description="Estilo editorial dos cortes.")


class ClipSuggestion(BaseModel):
    start: float = Field(description="Segundo exato de início do corte.")
    end: float = Field(description="Segundo exato de término do corte (duração entre 30 e 60 segundos).")
    title: str = Field(description="Título chamativo e natural em português com até 72 caracteres.")
    reason: str = Field(description="Justificativa objetiva em português (1 a 2 frases) explicando por que o trecho funciona.")


class ClipSelectionResponse(BaseModel):
    clips: list[ClipSuggestion]


# Armazenamento em memória do estado de cada job ativo
JOBS: dict[str, dict[str, Any]] = {}
JOB_EVENTS: dict[str, asyncio.Event] = {}


# ---------------------------------------------------------------------------
# Utilitários: extração de ID, limpeza de disco e atualização SSE
# ---------------------------------------------------------------------------
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(raw_url: str) -> str:
    """Extrai e valida o ID de 11 caracteres de links padrão, live, shorts ou youtu.be."""
    url = raw_url.strip()
    if not url:
        raise ValueError("Cole um link válido do YouTube antes de continuar.")

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().replace("www.", "").replace("m.", "")
    video_id = ""

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(parts) >= 2 and parts[0] in {"live", "shorts", "embed", "v"}:
                video_id = parts[1]

    if not YOUTUBE_ID_RE.match(video_id):
        raise ValueError(
            "Link do YouTube inválido. Use um endereço como "
            "https://www.youtube.com/watch?v=... ou https://youtu.be/..."
        )
    return video_id


def cleanup_old_jobs() -> None:
    """Remove diretórios de jobs antigos em /tmp para evitar lotação no plano gratuito."""
    now = time.time()
    expired_ids = [
        job_id
        for job_id, state in list(JOBS.items())
        if now - state.get("created_at", now) > JOB_TTL_SECONDS
    ]
    for job_id in expired_ids:
        JOBS.pop(job_id, None)
        JOB_EVENTS.pop(job_id, None)
        shutil.rmtree(BASE_TEMP_DIR / job_id, ignore_errors=True)

    # Limpa eventuais diretórios órfãos deixados após reinicialização do container
    for folder in BASE_TEMP_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            age = now - folder.stat().st_mtime
            if age > JOB_TTL_SECONDS and folder.name not in JOBS:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: int | None = None,
    progress: int | None = None,
    message: str | None = None,
    clips: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    """Atualiza o estado do job e acorda clientes SSE conectados."""
    job = JOBS.get(job_id)
    if not job:
        return

    if status is not None:
        job["status"] = status
    if stage is not None:
        job["stage"] = stage
    if progress is not None:
        job["progress"] = progress
    if message is not None:
        job["message"] = message
    if clips is not None:
        job["clips"] = clips
    if error is not None:
        job["error"] = error
    job["updated_at"] = time.time()

    event = JOB_EVENTS.get(job_id)
    if event:
        event.set()


# ---------------------------------------------------------------------------
# Etapa 1: Extração da transcrição com timestamps (youtube-transcript-api)
# ---------------------------------------------------------------------------
def _extract_snippet_fields(snippet: Any) -> tuple[float, float, str]:
    """Compatível com youtube-transcript-api v1.x (objetos) e v0.6.x (dicionários)."""
    if isinstance(snippet, dict):
        start = float(snippet.get("start", 0.0))
        duration = float(snippet.get("duration", 0.0))
        text = str(snippet.get("text", "")).replace("\n", " ").strip()
        return start, duration, text
    start = float(getattr(snippet, "start", 0.0))
    duration = float(getattr(snippet, "duration", 0.0))
    text = str(getattr(snippet, "text", "")).replace("\n", " ").strip()
    return start, duration, text


def fetch_transcript_sync(video_id: str) -> tuple[str, float, str]:
    """Baixa a transcrição priorizando português ou qualquer idioma disponível."""
    preferred_pt = ["pt", "pt-BR", "pt-PT"]
    preferred_fallback = ["en", "es"]

    try:
        ytt_api = YouTubeTranscriptApi()
        if hasattr(ytt_api, "list"):
            transcript_list = ytt_api.list(video_id)
        else:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]

        selected = None
        for langs in (preferred_pt, preferred_fallback):
            try:
                selected = transcript_list.find_transcript(langs)
                break
            except NoTranscriptFound:
                continue

        if selected is None:
            available = list(transcript_list)
            if not available:
                raise NoTranscriptFound(video_id, preferred_pt, transcript_list)
            # Prefere legenda manual se existir; caso contrário usa a primeira gerada
            manual = [t for t in available if not getattr(t, "is_generated", True)]
            selected = manual[0] if manual else available[0]

        fetched = selected.fetch()
        lang_code = getattr(selected, "language_code", "desconhecido")

    except TranscriptsDisabled as exc:
        raise RuntimeError(
            "Este vídeo tem as legendas desativadas pelo criador. Escolha uma live ou vídeo com legendas ativadas."
        ) from exc
    except NoTranscriptFound as exc:
        raise RuntimeError(
            "Nenhuma legenda ou transcrição automática foi encontrada para este vídeo."
        ) from exc
    except VideoUnavailable as exc:
        raise RuntimeError(
            "Vídeo indisponível, privado ou removido no YouTube."
        ) from exc
    except Exception as exc:
        err_name = exc.__class__.__name__.lower()
        err_msg = str(exc).lower()
        if "blocked" in err_name or "ipblocked" in err_name or "too many requests" in err_msg:
            raise RuntimeError(
                "O YouTube bloqueou temporariamente a leitura de legendas deste servidor. Tente novamente em alguns minutos."
            ) from exc
        if "age" in err_name or "age" in err_msg:
            raise RuntimeError(
                "Este vídeo possui restrição de idade e não permite leitura automática de transcrição."
            ) from exc
        raise RuntimeError(f"Não foi possível obter a transcrição do vídeo: {exc}") from exc

    snippets = list(fetched)
    if not snippets:
        raise RuntimeError("A transcrição retornada pelo YouTube está vazia.")

    # Agrupa frases curtas em blocos de ~12 segundos para reduzir ruído e economizar tokens
    blocks: list[tuple[int, str]] = []
    block_start = -1
    block_texts: list[str] = []
    max_end = 0.0

    for item in snippets:
        start, duration, text = _extract_snippet_fields(item)
        max_end = max(max_end, start + max(duration, 1.0))
        if not text:
            continue
        if block_start < 0:
            block_start = int(start)
            block_texts = [text]
        elif start - block_start >= 12:
            blocks.append((block_start, " ".join(block_texts)))
            block_start = int(start)
            block_texts = [text]
        else:
            block_texts.append(text)

    if block_start >= 0 and block_texts:
        blocks.append((block_start, " ".join(block_texts)))

    if max_end < 35:
        raise RuntimeError(
            "O vídeo é muito curto (menos de 35 segundos) para gerar cortes de 30 a 60 segundos."
        )

    lines = [f"[{sec}s] {txt}" for sec, txt in blocks]
    formatted = "\n".join(lines)

    # Em lives extremamente longas (+5h), condensa linhas proporcionalmente sem perder o arco temporal
    if len(formatted) > MAX_TRANSCRIPT_CHARS:
        step = max(2, len(lines) // (MAX_TRANSCRIPT_CHARS // 95))
        formatted = "\n".join(lines[::step])[:MAX_TRANSCRIPT_CHARS]

    return formatted, max_end, lang_code


# ---------------------------------------------------------------------------
# Etapa 2: Seleção dos melhores trechos com Gemini 2.5 Flash (JSON estrito)
# ---------------------------------------------------------------------------
STYLE_DIRECTIONS: dict[ClipStyle, str] = {
    "viral": (
        "Priorize ganchos fortes nos primeiros 3 segundos, opiniões polêmicas ou contraintuitivas, "
        "reviravoltas, histórias com alto poder de retenção e frases marcantes que prendem atenção em Shorts/Reels/TikTok."
    ),
    "educativo": (
        "Priorize explicações claras e autocontidas, dicas práticas acionáveis, passo a passo, "
        "insights técnicos ou aprendizados que entregam valor completo dentro do trecho sem depender do resto da live."
    ),
    "engracado": (
        "Priorize momentos espontâneos de humor, reações exageradas, histórias engraçadas, "
        "quebras de expectativa, sarcasmo e diálogos divertidos com timing cômico completo."
    ),
}


def select_clips_with_gemini_sync(
    transcript: str,
    total_duration: float,
    clip_count: int,
    style: ClipStyle,
) -> list[dict[str, Any]]:
    """Envia a transcrição para o gemini-2.5-flash exigindo JSON estruturado."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Chave GEMINI_API_KEY não encontrada. Configure-a em Settings → Variables and secrets no seu Hugging Face Space."
        )

    client = genai.Client(api_key=api_key)
    style_prompt = STYLE_DIRECTIONS.get(style, STYLE_DIRECTIONS["viral"])

    prompt = f"""Você é um editor sênior especialista em cortes verticais (Reels, TikTok e YouTube Shorts).
Analise a transcrição com timestamps em segundos abaixo e escolha EXATAMENTE {clip_count} melhores momentos do vídeo.

DIRETRIZ DE ESTILO ({style.upper()}):
{style_prompt}

REGRAS OBRIGATÓRIAS:
1. Cada corte DEVE durar entre 30 e 60 segundos (end - start >= 30 e end - start <= 60).
2. Todos os timestamps devem estar dentro do intervalo real do vídeo: 0 até {int(total_duration)} segundos.
3. Os cortes NÃO podem se sobrepor entre si.
4. Comece cada corte no início de uma frase natural e termine após a conclusão da ideia (sem cortar no meio de uma palavra).
5. Escreva `title` (máximo 72 caracteres, direto e sem hashtags) e `reason` (1 a 2 frases objetivas) em português do Brasil.
6. Retorne estritamente no formato JSON definido pelo schema.

TRANSCRIÇÃO (formato [segundos] texto):
{transcript}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClipSelectionResponse,
                temperature=0.4,
            ),
        )
    except Exception as exc:
        err_text = str(exc)
        if "API_KEY_INVALID" in err_text or "403" in err_text or "401" in err_text:
            raise RuntimeError(
                "A chave GEMINI_API_KEY configurada parece inválida ou sem permissão. Verifique-a nos Secrets do Hugging Face."
            ) from exc
        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
            raise RuntimeError(
                "Limite de requisições da API do Gemini atingido temporariamente. Aguarde alguns instantes e tente novamente."
            ) from exc
        raise RuntimeError(f"Erro ao consultar o Gemini 2.5 Flash: {exc}") from exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("A IA retornou uma resposta vazia ao analisar a transcrição.")

    try:
        parsed = ClipSelectionResponse.model_validate_json(raw_text)
    except Exception:
        data = json.loads(raw_text)
        clips_raw = data.get("clips", data if isinstance(data, list) else [])
        parsed = ClipSelectionResponse(clips=[ClipSuggestion(**c) for c in clips_raw])

    if not parsed.clips:
        raise RuntimeError("A IA não identificou momentos elegíveis neste vídeo.")

    # Sanitiza e garante duração entre 30s e 60s dentro da duração total do vídeo
    normalized: list[dict[str, Any]] = []
    for item in parsed.clips:
        start = max(0.0, round(float(item.start), 1))
        end = min(total_duration, round(float(item.end), 1))

        if end <= start:
            continue

        duration = end - start
        if duration < 30.0:
            end = min(total_duration, start + 35.0)
            if end - start < 30.0:
                start = max(0.0, end - 30.0)
        elif duration > 60.0:
            end = round(start + 60.0, 1)

        final_duration = round(end - start, 1)
        if final_duration < 15.0:
            continue

        # Evita cortes duplicados/sobrepostos
        overlaps = any(
            not (end <= existing["start"] + 2 or start >= existing["end"] - 2)
            for existing in normalized
        )
        if overlaps:
            continue

        normalized.append(
            {
                "start": start,
                "end": end,
                "duration": final_duration,
                "title": item.title.strip()[:90] or f"Corte #{len(normalized) + 1}",
                "reason": item.reason.strip()[:300] or "Momento de destaque selecionado pela IA.",
            }
        )
        if len(normalized) >= clip_count:
            break

    if not normalized:
        raise RuntimeError(
            "Os timestamps sugeridos pela IA não eram válidos para a duração deste vídeo. Tente novamente."
        )

    normalized.sort(key=lambda c: c["start"])
    return normalized


# ---------------------------------------------------------------------------
# Etapa 3: Download parcial com yt-dlp + recorte vertical 9:16 com FFmpeg
# ---------------------------------------------------------------------------
async def run_subprocess(cmd: list[str], timeout: int = 240) -> tuple[int, str]:
    """Executa subprocesso de forma assíncrona sem travar o loop SSE do FastAPI."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("Tempo limite excedido ao baixar ou converter um trecho do vídeo.")

    output = (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
    return proc.returncode or 0, output


async def cut_and_crop_clip(
    video_url: str,
    start: float,
    end: float,
    job_dir: Path,
    clip_index: int,
) -> Path:
    """Baixa apenas o intervalo escolhido via yt-dlp --download-sections e converte para 9:16 limpo."""
    raw_template = job_dir / f"raw_{clip_index}.%(ext)s"
    final_path = job_dir / f"corte_{clip_index:02d}.mp4"

    # Baixa apenas o trecho exato (até 1080p) economizando RAM, CPU e banda no Hugging Face
    yt_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--quiet",
        "-f",
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]/best",
        "--merge-output-format",
        "mp4",
        "--download-sections",
        f"*{start:.2f}-{end:.2f}",
        "--force-keyframes-at-cuts",
        "-o",
        str(raw_template),
        video_url,
    ]

    code, err_out = await run_subprocess(yt_cmd, timeout=240)
    downloaded_candidates = sorted(job_dir.glob(f"raw_{clip_index}.*"))
    if code != 0 or not downloaded_candidates:
        detail = err_out.splitlines()[-1] if err_out else "Erro desconhecido no yt-dlp."
        raise RuntimeError(f"Falha ao baixar trecho {clip_index} do YouTube: {detail}")

    source_file = downloaded_candidates[0]

    # Recorte centralizado 9:16 seguro para qualquer proporção de origem + 720x1280 rápido na CPU
    vf_filter = (
        "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,"
        "scale=720:1280:flags=lanczos,"
        "setsar=1"
    )

    ff_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_file),
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(final_path),
    ]

    try:
        ff_code, ff_err = await run_subprocess(ff_cmd, timeout=240)
        if ff_code != 0 or not final_path.exists():
            detail = ff_err.splitlines()[-1] if ff_err else "Erro desconhecido no FFmpeg."
            raise RuntimeError(f"Falha ao converter corte {clip_index} para 9:16: {detail}")
    finally:
        # Remove o trecho bruto imediatamente para economizar disco
        for temp_file in downloaded_candidates:
            temp_file.unlink(missing_ok=True)

    return final_path


# ---------------------------------------------------------------------------
# Pipeline principal executado em background
# ---------------------------------------------------------------------------
async def process_job(job_id: str, video_id: str, clip_count: int, style: ClipStyle) -> None:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    job_dir = BASE_TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Etapa 1: Lendo transcrição
        update_job(
            job_id,
            status="running",
            stage=1,
            progress=12,
            message="Etapa 1: Lendo transcrição do vídeo no YouTube...",
        )
        transcript_text, total_duration, lang_code = await asyncio.to_thread(
            fetch_transcript_sync, video_id
        )

        # Etapa 2: IA escolhendo melhores momentos
        update_job(
            job_id,
            stage=2,
            progress=32,
            message=f"Etapa 2: Gemini 2.5 Flash analisando transcrição ({lang_code}) no estilo {style}...",
        )
        suggestions = await asyncio.to_thread(
            select_clips_with_gemini_sync,
            transcript_text,
            total_duration,
            clip_count,
            style,
        )

        # Etapa 3: Cortando vídeos
        total_clips = len(suggestions)
        ready_clips: list[dict[str, Any]] = []
        failed_errors: list[str] = []

        for idx, clip in enumerate(suggestions, start=1):
            base_progress = 45 + int(((idx - 1) / total_clips) * 48)
            update_job(
                job_id,
                stage=3,
                progress=base_progress,
                message=f"Etapa 3: Cortando vídeo {idx} de {total_clips} para formato vertical 9:16...",
            )
            try:
                output_file = await cut_and_crop_clip(
                    video_url=video_url,
                    start=clip["start"],
                    end=clip["end"],
                    job_dir=job_dir,
                    clip_index=idx,
                )
                ready_clips.append(
                    {
                        "id": f"clip-{idx}",
                        "index": idx,
                        "title": clip["title"],
                        "reason": clip["reason"],
                        "start": clip["start"],
                        "end": clip["end"],
                        "duration": clip["duration"],
                        "video_url": f"/api/clips/{job_id}/{output_file.name}",
                        "download_url": f"/api/clips/{job_id}/{output_file.name}?download=1",
                    }
                )
            except Exception as clip_err:
                failed_errors.append(str(clip_err))

        if not ready_clips:
            reason = failed_errors[0] if failed_errors else "Não foi possível gerar nenhum corte."
            raise RuntimeError(reason)

        # Etapa 4: Pronto
        update_job(
            job_id,
            status="done",
            stage=4,
            progress=100,
            message=f"Etapa 4: Pronto! {len(ready_clips)} corte(s) vertical(is) 9:16 gerados com sucesso.",
            clips=ready_clips,
        )

    except Exception as exc:
        update_job(
            job_id,
            status="error",
            message="Ocorreu um erro durante o processamento.",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Rotas HTTP e Server-Sent Events (SSE)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> FileResponse:
    index_path = APP_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado.")
    return FileResponse(index_path, media_type="text/html")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }


@app.post("/api/jobs")
async def create_job(payload: CreateJobRequest) -> dict[str, str]:
    cleanup_old_jobs()

    try:
        video_id = extract_video_id(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise HTTPException(
            status_code=500,
            detail=(
                "Variável de ambiente GEMINI_API_KEY não configurada no servidor. "
                "Adicione sua chave em Settings → Variables and secrets no Hugging Face Spaces."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "video_id": video_id,
        "status": "running",
        "stage": 1,
        "progress": 5,
        "message": "Iniciando processamento da live...",
        "clips": [],
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    JOB_EVENTS[job_id] = asyncio.Event()

    asyncio.create_task(process_job(job_id, video_id, payload.clip_count, payload.style))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Processamento não encontrado ou expirado.")

    async def event_generator():
        last_sent_at = 0.0
        while True:
            job = JOBS.get(job_id)
            if not job:
                payload = json.dumps({"status": "error", "error": "Sessão expirada no servidor."})
                yield f"data: {payload}\n\n"
                break

            if job["updated_at"] != last_sent_at:
                last_sent_at = job["updated_at"]
                snapshot = {
                    "status": job["status"],
                    "stage": job["stage"],
                    "progress": job["progress"],
                    "message": job["message"],
                    "clips": job["clips"],
                    "error": job["error"],
                }
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"

                if job["status"] in {"done", "error"}:
                    break

            event = JOB_EVENTS.get(job_id)
            if not event:
                break

            event.clear()
            try:
                # Envia heartbeat SSE a cada 15s para manter conexão aberta no proxy do Hugging Face
                await asyncio.wait_for(event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/clips/{job_id}/{filename}")
async def serve_clip(
    job_id: str,
    filename: str,
    download: int = Query(default=0),
) -> FileResponse:
    # Proteção contra path traversal
    if not re.match(r"^[a-f0-9]{12}$", job_id) or not re.match(r"^corte_\d{2}\.mp4$", filename):
        raise HTTPException(status_code=400, detail="Identificador de corte inválido.")

    file_path = (BASE_TEMP_DIR / job_id / filename).resolve()
    if not str(file_path).startswith(str(BASE_TEMP_DIR.resolve())) or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Corte não encontrado ou já removido pela limpeza automática.",
        )

    headers = {}
    if download == 1:
        headers["Content-Disposition"] = f'attachment; filename="corteflow_{job_id}_{filename}"'

    return FileResponse(path=file_path, media_type="video/mp4", headers=headers)
