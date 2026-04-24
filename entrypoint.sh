#!/bin/bash
set -e

echo "⏳ Waiting for Qdrant to be ready..."
until curl -sf http://qdrant:6333/healthz > /dev/null; do
  sleep 2
done
echo "✅ Qdrant is ready."

# Check if collection already has data
POINTS=$(curl -s http://qdrant:6333/collections/medquad_qa | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('result', {}).get('points_count', 0))
except:
    print(0)
")

if [ "$POINTS" -lt 1000 ]; then
  echo "📂 Collection empty or missing. Running indexer to load MedQuAD XML data..."
  QDRANT_URL=http://qdrant:6333 python3 indexer.py
  echo "✅ Indexing complete."
else
  echo "✅ Collection already has $POINTS documents. Skipping indexing."
fi

echo "🚀 Starting FastAPI RAG API on port 8000..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
