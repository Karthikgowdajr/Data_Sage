FROM python:3.11-slim

WORKDIR /app

# Install system deps (safe minimal set)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements only
COPY requirements.backend.txt .

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements.backend.txt

# Copy backend code
COPY app ./app

# Expose port (Render uses $PORT internally)
EXPOSE 10000

# Start FastAPI
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "10000"]
