# syntax=docker/dockerfile:1
# The digest pins the multi-platform official Python image inspected for this recipe.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
FROM ${PYTHON_IMAGE} AS builder

ARG TARGETARCH
RUN test "$TARGETARCH" = arm64
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && /opt/venv/bin/python -m pip check

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 elvoice \
    && useradd --uid 10001 --gid elvoice --create-home elvoice \
    && mkdir -p /workspace /models /data \
    && chown elvoice:elvoice /workspace /models /data

COPY --from=builder /opt/venv /opt/venv
COPY tools/check_environment.py tools/download_models.py models.json /opt/el-voice/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HF_HOME=/data/huggingface \
    XDG_CACHE_HOME=/data/cache \
    OMP_NUM_THREADS=6 \
    OPENBLAS_NUM_THREADS=6 \
    EL_VOICE_MODELS=/models
USER elvoice
WORKDIR /workspace
# Fail the build if any of the native inference libraries cannot be imported.
RUN python /opt/el-voice/check_environment.py --skip-models
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).read()"]
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
