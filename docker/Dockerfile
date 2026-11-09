ARG BASE_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

FROM ${UV_IMAGE} AS uv

FROM ${BASE_IMAGE}

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev tini curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --frozen --group gpu --no-dev

COPY models/ /models/
COPY licenses/ /licenses/

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY docker/ /app/docker/

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HOME=/tmp/hf_home \
    VLLM_CACHE_ROOT=/tmp/vllm_cache \
    TRITON_CACHE_DIR=/tmp/triton_cache \
    XDG_CACHE_HOME=/tmp/xdg_cache \
    LEGALMT_ARTIFACTS_ROOT=/tmp/legalmt \
    LEGALMT_SERVE_CONFIG=/app/configs/LegalMT/serve.image.yaml \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"]

LABEL org.opencontainers.image.title="LegalMT VLSP 2026 translation server" \
      org.opencontainers.image.description="Offline EN<->VI legal document translation over HTTP (POST /translate)" \
      org.opencontainers.image.licenses="see /licenses"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/.venv/bin/python", "/app/docker/server.py"]
