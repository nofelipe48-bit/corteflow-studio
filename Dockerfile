FROM python:3.10-slim

# Evita arquivos .pyc e garante logs em tempo real no painel do Hugging Face Spaces
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Instala ffmpeg e certificados SSL necessários para yt-dlp e chamadas HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root com UID 1000 (exigido pelo ambiente Docker do Hugging Face Spaces)
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Copia e instala dependências antes do código para aproveitar cache de camadas do Docker
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia os arquivos da aplicação com permissão do usuário 1000
COPY --chown=user:user app.py index.html ./

# Cria pasta temporária gravável para os cortes
RUN mkdir -p /tmp/corteflow_jobs && chown -R user:user /tmp/corteflow_jobs

USER user

# Porta padrão obrigatória do Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
