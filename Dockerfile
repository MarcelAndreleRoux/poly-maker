# syntax=docker/dockerfile:1
# polymaker runtime image. Build:  docker compose build
# State (state.db, journal/, logs/) lives under /data, a bind mount from ./data on the host.
# Config is bind-mounted at /data/config so every CLI default path (config/, state.db,
# journal, logs) resolves without flags.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# dependency layer (cached until pyproject/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# project layer
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

ENV PATH="/app/.venv/bin:$PATH"

# working dir = data dir; config/ is mounted here by compose
RUN mkdir -p /data/config
WORKDIR /data

ENTRYPOINT ["polymaker"]
CMD ["--help"]
