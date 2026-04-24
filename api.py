"""
MedQuAD RAG API — FastAPI service
Exposes the RAG retriever as a REST API endpoint.
All answers come verbatim from the MedQuAD XML dataset (no LLM hallucination).
"""
import os
import time
from pathlib import Path
from typing import Annotated
from contextlib import asynccontextmanager

import qdrant_client
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA


# ── Shared state (loaded once at startup) ───────────────────────────────────
class RAGState:
    retriever = None


rag = RAGState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retriever once at startup — shared across all requests."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")

    embed_model = NVIDIAEmbedding(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=api_key,
        truncate="END"
    )
    llm = NVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=api_key,
        temperature=0.1
    )
    Settings.embed_model = embed_model
    Settings.llm = llm

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key_qdrant = os.environ.get("QDRANT_API_KEY")
    qdrant_c = qdrant_client.QdrantClient(url=qdrant_url, api_key=api_key_qdrant)
    vector_store = QdrantVectorStore(
        client=qdrant_c,
        collection_name="medquad_qa",
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        sparse_vector_name="text-sparse-new",  # Must match the name used during indexing
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    rag.retriever = index.as_retriever(
        similarity_top_k=5,
        vector_store_query_mode="hybrid",   # Use RRF to fuse dense + sparse results
    )
    print("✅ RAG retriever initialized and ready.")
    yield
    print("Shutting down RAG API...")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedQuAD RAG API",
    description="Medical Q&A from the MedQuAD dataset. Answers come verbatim from NIH XML files — no LLM generation.",
    version="1.0.0",
    lifespan=lifespan
)


# ── Response Models ──────────────────────────────────────────────────────────
class RetrievedDocument(BaseModel):
    rank: int
    source: str
    focus: str
    url: str
    relevance_score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    retrieved_in_seconds: float
    results: list[RetrievedDocument]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    index = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=index.read_text())


@app.get("/health")
def health() -> dict:
    ready = rag.retriever is not None
    return {"status": "ok" if ready else "initializing", "retriever_ready": ready}


@app.get("/query", response_model=QueryResponse)
def query_medquad(
    q: Annotated[str, Query(min_length=3, description="Your medical question")],
    top_k: Annotated[int, Query(ge=1, le=20, description="Number of results to return")] = 5,
) -> QueryResponse:
    """
    Query the MedQuAD XML knowledge base.
    Returns verbatim text from NIH medical documents — no LLM synthesis.
    """
    if rag.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready yet.")

    # Adjust top_k dynamically if needed
    rag.retriever._similarity_top_k = top_k

    start = time.time()
    nodes = rag.retriever.retrieve(q)
    elapsed = round(time.time() - start, 3)

    results = [
        RetrievedDocument(
            rank=i + 1,
            source=node.metadata.get("source", "Unknown"),
            focus=node.metadata.get("focus", "Unknown"),
            url=node.metadata.get("url", "N/A"),
            relevance_score=round(node.score or 0.0, 4),
            text=node.get_text()
        )
        for i, node in enumerate(nodes)
    ]

    return QueryResponse(query=q, retrieved_in_seconds=elapsed, results=results)
