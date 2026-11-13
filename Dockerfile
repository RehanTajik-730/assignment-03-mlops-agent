# Production container image for MLOps FastAPI Inference Service
FROM python:3.11-slim

# Set working directory and non-root user for security
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    APP_VERSION=v1.4.2

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code and model artifacts
COPY api/ ./api/
COPY models/ ./models/

# Expose HTTP port
EXPOSE 8000

# Health check matching Kubernetes readiness probe
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health/ready || exit 1

# Run inference server with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
