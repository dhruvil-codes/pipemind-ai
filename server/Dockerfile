# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy the full environment package
COPY . .

# Install the package itself (for relative imports to work)
RUN pip install --no-cache-dir -e .

# Environment config
ENV TASK_ID=easy
ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "pipeline_debugger_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
