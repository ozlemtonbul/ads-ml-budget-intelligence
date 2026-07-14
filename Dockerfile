FROM python:3.13-slim

# Python settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY config ./config
COPY src ./src
COPY main.py .
COPY README.md .
COPY .env.example .

# Create required folders
RUN mkdir -p /app/outputs \
    && mkdir -p /app/logs

# Default command
CMD ["python", "main.py"]