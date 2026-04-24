FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed BM25 model so it is cached in the Docker image
RUN python -c "from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"

# Copy ONLY the API code (we don't need the 1GB of XML data anymore as it's in Qdrant Cloud)
COPY api.py ./
COPY static ./static

EXPOSE 8080

# Run with uvicorn directly
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
