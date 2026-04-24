import os
import time
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

px.launch_app()
tracer_provider = register(project_name="medquad-rag", auto_instrument=False)
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

def test_inference_speed():
    # Only test Nvidia since DeepInfra API key hasn't been provided yet
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY is not set.")
        return
        
    try:
        from llama_index.llms.nvidia import NVIDIA
        
        print("--- Testing Nvidia NIM API (qwen2.5-7b-instruct) ---")
        
        # Initialize the NIM LLM
        # The Qwen 2.5 7B model hosted on NVIDIA NIM
        llm = NVIDIA(model="qwen/qwen2.5-7b-instruct", api_key=api_key)
        
        prompt = "Explain Retrieval-Augmented Generation (RAG) in 3 short sentences."
        
        start_time = time.time()
        print(f"Sending prompt: '{prompt}'")
        response = llm.complete(prompt)
        end_time = time.time()
        
        duration = end_time - start_time
        
        print(f"\nResponse (took {duration:.2f} seconds):")
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_inference_speed()
