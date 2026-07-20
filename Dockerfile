# ============================================================
# Base Image
# ============================================================

FROM python:3.13-slim

# ============================================================
# Environment Variables
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ============================================================
# Working Directory
# ============================================================

WORKDIR /app

# ============================================================
# System Dependencies
# ============================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Python Dependencies
# ============================================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================================
# Copy Project Files
# ============================================================

COPY config ./config
COPY src ./src
COPY docs ./docs
COPY tests ./tests
COPY main.py .
COPY README.md .
COPY .env.example .

# ============================================================
# Create Runtime Directories
# ============================================================

RUN mkdir -p \
    /app/outputs \
    /app/logs

# ============================================================
# Default Command
# ============================================================

CMD ["python", "main.py"]