FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False ANONYMIZED_TELEMETRY=False \
    CLINICALCHAT_HOME=/app WORK_DIR=/app/data LLM_PROVIDER=openai PORT=7860
WORKDIR /app

# libgl/libglib are needed by the OCR stack (rapidocr / opencv)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .
COPY data/raw ./data/raw

# parse the dataset + build the vector index at build time so the container starts fast
RUN LLM_PROVIDER=auto clinicalchat ingest

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:7860/health || exit 1
CMD ["clinicalchat", "serve"]
