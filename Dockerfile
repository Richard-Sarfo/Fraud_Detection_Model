# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1

# libgomp1 is the OpenMP runtime that xgboost/lightgbm wheels link against.
# build-essential is intentionally omitted — every dep in requirements.txt
# ships a manylinux wheel for cpython-3.11/x86_64, so no source compilation
# is needed. This drops ~250 MB and ~30s from every fresh build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# BuildKit cache mount: pip's wheel cache persists across builds, so repeat
# builds reuse downloaded wheels instead of fetching them again. This is the
# single largest win for incremental rebuilds (e.g. after editing
# requirements.txt or pruning the docker cache).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

EXPOSE 8000 8888

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
