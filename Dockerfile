# EntroSight application container
# Python 3.11 slim base, CPU-only PyTorch (no GPU required)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies:
# - build-essential: compile any packages without wheels
# - libgomp1: OpenMP runtime required by PyTorch CPU
# - libjpeg62-turbo, zlib1g, libpng16-16: Pillow image codecs (JPEG/zlib/PNG)
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        libgomp1 \
        libjpeg62-turbo \
        zlib1g \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first from the dedicated CPU index so the image stays
# small and GPU-free. Installing these before requirements.txt ensures the
# unpinned torch/torchvision entries there are already satisfied by the CPU build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch \
        torchvision

# Install the remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, RAG knowledge base, and model checkpoints
COPY app/ ./app/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY models/ ./models/

# Create runtime data directories (generated at runtime, not committed)
RUN mkdir -p data/chromadb data/heatmaps

# FastAPI/Uvicorn service port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
