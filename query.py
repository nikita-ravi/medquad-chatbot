import os
import time
import qdrant_client
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from langsmith import traceable
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from langsmith import Client

px.launch_app()
tracer_provider = register(project_name="medquad-rag", auto_instrument=False)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Ensure LangSmith captures the runs globally
os.environ["LANGCHAIN_TRACING_V2"] = "true"

@traceable(name="MedQuAD RAG Query")
def query_rag_system(user_query: str):
    # 1. Check API Keys
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is not set.")
    
    # 2. Configure Models
    # Set the same Nvidia Embedder used during indexing
    embed_model = NVIDIAEmbedding(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=api_key,
        truncate="END"
    )
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    
    # Use Nvidia's API to access Llama 3.1 (Falling back since Qwen NIM was timing out)
    llm = NVIDIA(
        model="meta/llama-3.1-8b-instruct", 
        api_key=api_key,
        temperature=0.1
    )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # 3. Connect to existing Qdrant Vector Store
    print("Connecting to Qdrant (http://localhost:6333)...")
    qdrant_c = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        client=qdrant_c, 
        collection_name="medquad_qa"
    )
    
    # 5. Use retriever only — NO LLM synthesis, answers come directly from XML text
    print("Initializing Retriever (XML-only mode, no LLM)...")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=5)

    start_time = time.time()
    print(f"\nQuerying: '{user_query}'...")
    retrieved_nodes = retriever.retrieve(user_query)
    end_time = time.time()

    # 6. Display raw XML text — no LLM involved
    print("\n" + "="*60)
    print("ANSWER FROM MedQuAD XML (NO LLM — RAW RETRIEVED TEXT)")
    print("="*60)

    for i, node in enumerate(retrieved_nodes):
        metadata = node.metadata
        score = node.score if node.score else 0.0
        focus = metadata.get('focus', 'Unknown')
        source = metadata.get('source', 'Unknown')
        url = metadata.get('url', 'N/A')
        text = node.get_text()

        print(f"\n[{i+1}] Source: {source} | Focus: {focus} | Relevance: {score:.4f}")
        print(f"URL: {url}")
        print(f"\n{text}\n")
        print("-"*60)

    print(f"\n[Retrieved in {end_time - start_time:.2f} seconds — text above is verbatim from XML]")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the MedQuAD Medical RAG System!")
    print("Type 'exit' or 'quit' to close.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter your medical question: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
                
            if user_input.strip() == "":
                continue
                
            query_rag_system(user_input)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error executing query: {e}")
