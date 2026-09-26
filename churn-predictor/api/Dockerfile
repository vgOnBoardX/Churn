# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy python packages from builder
COPY --from=builder /install /usr/local

# Copy API app code
COPY app/ ./app/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DATABASE_URL="sqlite+aiosqlite:///./churn.db" \
    MODEL_PATH="/app/models/churn_pipeline.joblib" \
    METRICS_PATH="/app/models/metrics.json"

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
