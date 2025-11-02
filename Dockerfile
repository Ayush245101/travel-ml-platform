# Build stage (optional for caching)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src /app/src
COPY api /app/api
COPY models /app/models
COPY .env.example /app/.env.example

# Default environment
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

EXPOSE 8080
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "api.app:app"]