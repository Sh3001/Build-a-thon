# ---- stage 1: build the dashboard -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the application ------------------------------------------------
FROM python:3.11-slim
# `development` is the container default so `docker compose up` works with no secrets.
# The production profile refuses to start without them -- see backend/app/profiles.py.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RECOVERAI_PROFILE=development
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/
COPY simulation/ ./simulation/
COPY scripts/ ./scripts/
# The simulator scenarios, the cost model and the effect priors. These are inputs to
# every decision the system makes, so leaving them out would make the container behave
# differently from a local run while looking identical.
COPY config/ ./config/
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN pip install -e .

# Run as a non-root user. The application writes only to /app/data, which the volume
# mount owns, so nothing needs write access to the code.
RUN useradd --create-home --uid 10001 recoverai \
    && mkdir -p /app/data && chown -R recoverai:recoverai /app
USER recoverai

EXPOSE 8000
# Build the dataset, model and experiment on first boot, then serve.
CMD ["sh", "-c", "python scripts/demo.py --no-serve && exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
