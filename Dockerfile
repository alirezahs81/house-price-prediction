# Dockerfile for House Price Prediction
# Build: docker build -t house-price-prediction .
# Run training: docker run --rm house-price-prediction train
# Run prediction: docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports house-price-prediction predict

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY main.py .
COPY README.md .
COPY LICENSE .

# Create necessary directories
RUN mkdir -p /app/models /app/reports/figures /app/reports/metrics /app/data/processed

# Default command shows help
CMD ["python", "-m", "main", "--help"]