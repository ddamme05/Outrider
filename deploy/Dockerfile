# Keyless demo image — runs `outrider.main:app` in DEMO_MODE. No live LLM/GitHub/
# Slack credentials are baked in or needed; the app early-returns after the read-side
# wiring (see src/outrider/api/lifespan.py `if demo_mode:`).
FROM python:3.13.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer (cached on lockfile changes only).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# App sources only — deliberately NOT the repo root, so no .env / secrets / dashboard
# get copied into the image. hatchling builds the wheel from src/outrider.
COPY src/ ./src/
COPY db/ ./db/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

EXPOSE 8000
# DEMO_MODE is selected at runtime via OUTRIDER_DEMO_MODE=1 (set in compose).
CMD ["uv", "run", "uvicorn", "outrider.main:app", "--host", "0.0.0.0", "--port", "8000"]
