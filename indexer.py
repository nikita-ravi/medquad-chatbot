import os
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
import qdrant_client

# Only launch Phoenix UI locally (skip in Docker to avoid port conflicts)
if os.environ.get("RUN_ENV") != "docker":
    px.launch_app()
tracer_provider = register(project_name="medquad-rag", auto_instrument=False)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

from ingest import load_medquad_data

def build_index():
    # 1. Configure the Nvidia Embedding Model
    # Needs NVIDIA_API_KEY environment variable to be set
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it to use the Nvidia embedding model.")
    
    embed_model = NVIDIAEmbedding(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=api_key,
        truncate="END" # Truncate documents if they exceed context length
    )
    
    Settings.embed_model = embed_model
    
    # Optional: Configure Qwen LLM for later steps (retrieval/generation)
    # Settings.llm = ... (Will be added when needed)

    # 2. Setup Qdrant Vector Store
    # Connect to the local Qdrant Docker container
    print("Initializing Qdrant client...")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key_qdrant = os.environ.get("QDRANT_API_KEY")
    qdrant_c = qdrant_client.QdrantClient(url=qdrant_url, api_key=api_key_qdrant)
    
    vector_store = QdrantVectorStore(
        client=qdrant_c,
        collection_name="medquad_qa",
        enable_hybrid=True,               # Store sparse BM25 vectors alongside dense
        fastembed_sparse_model="Qdrant/bm25",  # BM25 sparse model via FastEmbed
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 3. Load Documents
    print("Loading documents from MedQuAD...")
    documents = load_medquad_data("./MedQuAD")
    
    # 4. Build Index (This process embeds all documents and stores them in Qdrant)
    print(f"Building index for {len(documents)} documents. This may take a while depending on rate limits...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True 
    )
    
    print("Indexing complete!")
    return index

if __name__ == "__main__":
    build_index()
