import os
import qdrant_client
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding

px.launch_app()
tracer_provider = register(project_name="medquad-rag", auto_instrument=False)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

api_key = os.environ.get("NVIDIA_API_KEY")

embed_model = NVIDIAEmbedding(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    api_key=api_key,
    truncate="END"
)

Settings.embed_model = embed_model
qdrant_c = qdrant_client.QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=qdrant_c, collection_name="medquad_qa")

index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What are the symptoms of Problems with Taste ?")
for i, node in enumerate(response.source_nodes):
    metadata = node.metadata
    print(f"[{i+1}] Source: {metadata.get('source', 'Unknown')} | Focus: {metadata.get('focus', 'Unknown')}")
    print(f"Preview: {node.get_text()[:150]}...\n")
