"""
Debug script: Shows raw retrieved text from Qdrant (no LLM involved at all).
This proves the data comes from the MedQuAD XML files, not from any LLM.
"""
import os
import qdrant_client
from llama_index.embeddings.nvidia import NVIDIAEmbedding

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY not set")

# Embed the query
embed_model = NVIDIAEmbedding(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    api_key=api_key,
    truncate="END"
)

query = input("Enter your question: ")
print("\n⏳ Embedding your query...\n")
query_embedding = embed_model.get_query_embedding(query)

# Search Qdrant directly — NO LLM
qdrant_c = qdrant_client.QdrantClient(url="http://localhost:6333")
response = qdrant_c.query_points(
    collection_name="medquad_qa",
    query=query_embedding,
    limit=5,
    with_payload=True
)
results = response.points

print("=" * 60)
print("RAW TEXT RETRIEVED FROM QDRANT (XML DATA — NO LLM)")
print("=" * 60)
for i, result in enumerate(results):
    payload = result.payload
    print(f"\n[{i+1}] Score: {result.score:.4f}")
    print(f"    Source:  {payload.get('source', 'N/A')}")
    print(f"    Focus:   {payload.get('focus', 'N/A')}")
    print(f"    URL:     {payload.get('url', 'N/A')}")
    # The actual XML text is in _node_content
    import json
    node_content = payload.get('_node_content', '{}')
    try:
        node = json.loads(node_content)
        text = node.get('text', payload.get('text', 'N/A'))
    except Exception:
        text = str(node_content)
    print(f"    RAW XML TEXT:\n    {text[:400]}")
    print("-" * 60)

print("\n✅ Above is the EXACT text from the XML files stored in Qdrant.")
print("   No LLM was used to generate any of the above.")
