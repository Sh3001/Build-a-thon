# ---- stage 1: build the dashboard -------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- stage 2: the application ------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ ./backend/
COPY ml/ ./ml/
COPY simulation/ ./simulation/
COPY scripts/ ./scripts/
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN pip install -e .

EXPOSE 8000
# Build the dataset, model and experiment on first boot, then serve.
CMD ["sh", "-c", "python scripts/demo.py --no-serve && exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
