# 🩺 MedQuAD RAG API

> A production-ready **Retrieval-Augmented Generation (RAG)** system for medical Q&A, powered by 17,000+ NIH documents. Answers are retrieved **verbatim** from source — no LLM hallucination.

**🌐 Live Demo** → [https://medquad-rag-api-243637556513.us-central1.run.app](https://medquad-rag-api-243637556513.us-central1.run.app)
**📖 API Docs** → [/docs](https://medquad-rag-api-243637556513.us-central1.run.app/docs)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Local Setup](#-local-setup)
- [Indexing Documents](#-indexing-documents)
- [API Reference](#-api-reference)
- [Cloud Deployment](#️-cloud-deployment)
- [Environment Variables](#-environment-variables)
- [How Hybrid RRF Search Works](#-how-hybrid-rrf-search-works)

---

## 🔍 Overview

MedQuAD RAG API is a medical question-answering service that uses Retrieval-Augmented Generation to search over the **MedQuAD (Medical Question Answer Dataset)** — a curated collection of Q&A pairs from NIH (National Institutes of Health) websites.

Instead of generating answers with an LLM (which can hallucinate), this system retrieves **exact, verbatim text** from validated medical documents. This makes it suitable for healthcare applications where accuracy and auditability matter.

**Example query:**
```
GET /query?q=What+are+the+symptoms+of+Type+2+Diabetes
```
**Returns** → The exact NIH-published answer about diabetes symptoms, with the source document, NIH URL, and relevance score.

---

## 🏗 Architecture

```
User Browser / API Client
        │
        ▼
┌─────────────────────────┐
│   FastAPI (Cloud Run)   │  ← Serves UI + REST API
│   + Beautiful Frontend  │
└────────────┬────────────┘
             │  Query
             ▼
┌─────────────────────────┐
│     LlamaIndex Core     │  ← Orchestrates retrieval
│   Hybrid RRF Retriever  │
└────┬──────────────┬─────┘
     │              │
     ▼              ▼
Dense Search    Sparse Search
NVIDIA Embed    FastEmbed BM25
     │              │
     └──────┬───────┘
            ▼
┌─────────────────────────┐
│      Qdrant Cloud       │  ← Hosted vector database
│  17,000+ indexed docs   │     (Free Tier)
└─────────────────────────┘
```

Retrieval flow:
1. User query is embedded using **NVIDIA `llama-3.2-nv-embedqa-1b-v2`** (dense vector)
2. The same query is tokenized using **FastEmbed BM25** (sparse vector)
3. Both results are fused using **Reciprocal Rank Fusion (RRF)**
4. Top-K results are returned as-is from the original NIH documents

---

## ✨ Features

- **🔀 Hybrid RRF Search** — Combines semantic (dense) and keyword (sparse/BM25) search for superior retrieval accuracy
- **📄 Verbatim Answers** — Returns exact text from NIH XML documents, no hallucination possible
- **⚡ Fast** — Typical retrieval in 1–3 seconds, even over 17,000+ documents
- **🌐 Beautiful Frontend** — Dark-themed medical UI with suggestion chips, result cards, and NIH source links
- **☁️ Serverless** — Deployed on Google Cloud Run (scales to zero, costs $0 at low traffic)
- **🔒 Secure** — All API keys injected via environment variables, never in source code
- **📊 Configurable** — Choose `top_k` (1–20 results) per query

---

## 📚 Dataset

This system is built on the **[MedQuAD dataset](https://github.com/abachaa/MedQuAD)**, a benchmark medical Q&A collection created from 12 NIH websites:

| Source | Description |
|---|---|
| NIDDK | National Institute of Diabetes and Digestive and Kidney Diseases |
| NCI | National Cancer Institute |
| NINDS | National Institute of Neurological Disorders and Stroke |
| NHLBI | National Heart, Lung, and Blood Institute |
| CDC | Centers for Disease Control and Prevention |
| MedlinePlus | NIH General Medical Encyclopedia |
| ...and more | 12 sources total |

**Stats:**
- **17,190+** Q&A document segments indexed
- Covers diseases, symptoms, treatments, genetics, clinical trials, and more

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | REST API + async web server |
| **RAG Orchestration** | LlamaIndex `0.14.x` | Retrieval pipeline management |
| **Dense Embeddings** | NVIDIA `llama-3.2-nv-embedqa-1b-v2` | Semantic vector search |
| **Sparse Embeddings** | FastEmbed `Qdrant/bm25` | Keyword-based BM25 search |
| **Vector Database** | Qdrant Cloud | Stores dense + sparse vectors |
| **Fusion Algorithm** | Reciprocal Rank Fusion (RRF) | Merges dense + sparse results |
| **Frontend** | Vanilla HTML/CSS/JS | Dark-themed UI (no framework needed) |
| **Container** | Docker | Reproducible runtime |
| **Cloud Hosting** | Google Cloud Run | Serverless container deployment |

---

## 📁 Project Structure

```
medquad-rag-api/
├── api.py                  # FastAPI application (main entry point)
├── indexer.py              # Script to ingest MedQuAD XML → Qdrant Cloud
├── ingest.py               # Alternative ingestion logic
├── query.py                # Standalone query script (for testing)
├── query_test.py           # Quick test queries
├── benchmark.py            # Performance benchmarking
├── debug_retrieval.py      # Debug retrieval results
├── static/
│   └── index.html          # Beautiful frontend UI
├── Dockerfile              # Cloud-optimized Docker image
├── Dockerfile.render       # Render.com alternative
├── docker-compose.yml      # Local dev with Qdrant
├── requirements.txt        # Python dependencies
├── render.yaml             # Render.com blueprint
├── entrypoint.sh           # Container startup script
├── .env.example            # Environment variable template
├── .gitignore              # Excludes secrets, data, venvs
└── README.md               # This file
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.10+
- A [Qdrant Cloud](https://cloud.qdrant.io) account (free tier)
- An [NVIDIA API Key](https://build.nvidia.com) (free tier, 1000 req/day)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/JAGAN666/medquad-rag-api.git
cd medquad-rag-api

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and fill in your NVIDIA_API_KEY, QDRANT_URL, QDRANT_API_KEY

# 5. Run the API
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** in your browser to see the UI.

> **Note:** The MedQuAD XML dataset is not included in this repo (1GB+). You need to download it from [github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD) and run the indexer before querying locally.

---

## 📥 Indexing Documents

Before the API can answer questions, you must index the MedQuAD XML files into Qdrant:

```bash
# Make sure your .env file has NVIDIA_API_KEY, QDRANT_URL, QDRANT_API_KEY
source .env  # or set env vars manually

python indexer.py
```

This will:
1. Walk all `*.xml` files in the `MedQuAD/` folder
2. Parse Q&A pairs and source metadata
3. Generate dense + sparse embeddings
4. Upload all vectors to your Qdrant Cloud collection (`medquad_qa`)

> ⏱️ **Estimated time:** ~15–25 minutes for 17,000+ documents (NVIDIA API rate limits apply)

---

## 📡 API Reference

### `GET /`
Returns the frontend web UI (HTML page).

---

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "retriever_ready": true
}
```

---

### `GET /query`
Query the MedQuAD knowledge base.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `q` | string | ✅ | — | Your medical question (min 3 chars) |
| `top_k` | integer | ❌ | `5` | Number of results to return (1–20) |

**Example Request:**
```bash
curl "https://medquad-rag-api-243637556513.us-central1.run.app/query?q=symptoms+of+asthma&top_k=3"
```

**Example Response:**
```json
{
  "query": "symptoms of asthma",
  "retrieved_in_seconds": 1.432,
  "results": [
    {
      "rank": 1,
      "source": "NHLBI",
      "focus": "Asthma",
      "url": "http://www.nhlbi.nih.gov/health/health-topics/topics/asth",
      "relevance_score": 6.312,
      "text": "Question: What are the symptoms of Asthma?\nAnswer: Common signs and symptoms of asthma include: coughing, especially at night, wheezing, chest tightness, shortness of breath..."
    }
  ]
}
```

---

### `GET /docs`
Interactive Swagger UI for exploring and testing all endpoints.

---

## ☁️ Cloud Deployment

### Google Cloud Run (recommended)

```bash
# 1. Install gcloud CLI
brew install --cask google-cloud-sdk

# 2. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Deploy (builds and pushes automatically)
gcloud run deploy medquad-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1024Mi \
  --timeout 300 \
  --set-env-vars "NVIDIA_API_KEY=YOUR_KEY,QDRANT_URL=YOUR_URL,QDRANT_API_KEY=YOUR_KEY,RUN_ENV=docker"
```

### Docker (local or any cloud)

```bash
# Build image
docker build -t medquad-rag-api .

# Run container
docker run -p 8080:8080 \
  -e NVIDIA_API_KEY=your_key \
  -e QDRANT_URL=your_url \
  -e QDRANT_API_KEY=your_key \
  -e RUN_ENV=docker \
  medquad-rag-api
```

### Render (one-click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/JAGAN666/medquad-rag-api)

> Fill in your `NVIDIA_API_KEY` when prompted, then click **Apply**.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | ✅ | Your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) |
| `QDRANT_URL` | ✅ | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | Your Qdrant Cloud API key |
| `RUN_ENV` | ❌ | Set to `docker` in containers (disables local Phoenix tracing) |

Copy `.env.example` to `.env` to get started:
```bash
cp .env.example .env
```

---

## 🔀 How Hybrid RRF Search Works

Traditional keyword search (BM25) is great for exact term matching but misses synonyms.
Semantic search (dense vectors) understands meaning but can miss specific medical terms.

**Reciprocal Rank Fusion (RRF)** combines both:

```
RRF_Score(doc) = Σ  1 / (k + rank_i)
                 i
```

Where `rank_i` is the document's rank in each individual ranking (dense or sparse), and `k=60` is a smoothing constant.

**Example:**
- Query: *"high glucose levels"*
- Dense search finds documents about *"hyperglycemia"* (semantic match)
- BM25 finds documents mentioning *"glucose"* literally (keyword match)
- RRF fuses both lists → best of both worlds

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  <p>Built with ❤️ using NVIDIA AI, LlamaIndex, Qdrant, and FastAPI</p>
  <p>Data from NIH MedQuAD — public domain medical knowledge</p>
</div>
