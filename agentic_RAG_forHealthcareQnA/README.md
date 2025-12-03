# realtime-healthcare

# Agentic RAG for Healthcare Question Answering

This repository implements a **deterministic, safety-aligned Agentic Retrieval-Augmented Generation (Agentic-RAG) system** designed for healthcare-oriented question answering. The system combines:

1. **Exact-match dataset lookup**
2. **Semantic retrieval using FAISS**
3. **An agentic controller that decides whether to answer from the dataset or call an LLM**
4. **Deterministic LLM fallback using HuggingFace or Gemini**
5. **A FastAPI-based microservice architecture with a public gateway**

The system prioritizes **accuracy**, **determinism**, **auditability**, and **safe fallback behavior**.
It never fabricates answers without evidence and falls back to *I don’t know* when appropriate.

---

## Overview of How the System Works

### 1. Data Source: Local Curated Healthcare QA Dataset

The dataset is loaded from:

```
data/healthcare_qa_dataset.jsonl
```

Each entry has:

```json
{
  "prompt": "What vaccinations do adults need?",
  "completion": "Adults typically require influenza..."
}
```

This dataset forms the **authoritative ground truth**.

---

### 2. Embedding and Retrieval (FAISS)

All dataset questions are embedded using:

```
sentence-transformers/all-MiniLM-L6-v2
```

A **FAISS inner-product index** is built and stored on disk:

* `rag_index.faiss`
* `rag_docs.json`
* `rag_embeddings.npy`

On each query:

1. The query is embedded.
2. FAISS returns the top-1 or top-k most similar dataset items.
3. The system computes:

   * cosine similarity (`score`)
   * lexical overlap (`frac`)
   * strict normalized exact-match

---

### 3. Agentic Controller

The controller decides the answer source:

* Return the dataset answer (if high-confidence)
* Call HuggingFace model
* Call Gemini model
* Abort safely with **I don’t know**

The decision is based on:

* Exact-match detection
* High similarity (≥ 0.8)
* Overlap fraction
* Multi-question detection
* Heuristic safety rules
* Optional controller-LLM advisory token

The controller is deterministic: dataset answers are always preferred when similarity and overlap are high.

---

### 4. LLM Call Path

The system supports two deterministic LLM backends:

#### HuggingFace Hub

Configured via:

```
HUGGINGFACEHUB_API_TOKEN
HUGGINGFACEHUB_REPO_ID
```

#### Google Gemini

Configured via:

```
GEMINI_API_KEY
GEMINI_MODEL
```

Call order:

1. Preferred backend (HF or Gemini based on controller)
2. Fallback backend
3. Final fallback → dataset
4. If nothing works → “I don’t know”

LLMs are used **only after retrieval fails**.

---

## File-by-File Explanation

### `rag_service.py`

The core RAG engine. Handles:

* Dataset loading
* FAISS indexing and retrieval
* Semantic similarity analysis
* Multi-question segmentation
* Exact-match normalization
* Agentic controller logic
* LLM fallback logic
* `/ask`, `/debug`, `/healthcheck` API routes

This is the main service running on port `8002`.

---

### `gateway_service.py`

The public-facing microservice:

* Handles requests from clients/frontends
* Forwards user queries to `rag_service`
* Applies rate limiting (optional)
* Provides clean JSON responses
* Runs on port `8000`

---

### `.env`

Stores required environment variables:

```
DATA_DIR=/data
GEMINI_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
HUGGINGFACEHUB_REPO_ID=...
GEMINI_MODEL=gemini-2.0-flash
```

These are automatically injected into Docker containers.

---

### `docker-compose.yml`

Brings up:

* `gateway-service`
* `rag-service`
* Shared volume for dataset and FAISS index

Automatically loads `.env`.

---

### `healthcare_qa_dataset.jsonl`

The curated primary dataset used by the RAG system.
Contains the authoritative QA knowledge base.

---

## Deterministic Behavior

This system is designed to be **predictable and reproducible**:

1. **Exact match → always dataset**
2. **High similarity + high overlap → dataset**
3. **Multi-question → forced LLM**
4. **Low overlap → forced LLM**
5. **No model available → dataset fallback or “I don’t know”**

LLMs are never used unless retrieval clearly fails.

---

## How to Run

### 1. Put your `.env` file in the project root

Example:

```
cp .env.example .env
```

### 2. Start services

```
docker compose up
```

### 3. Query the system

```
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What vaccinations do adults need?"}'
```

### 4. Debug environment

```
curl http://localhost:8002/debug
```

### 5. Healthcheck (FAISS, HF, Gemini)

```
curl http://localhost:8002/healthcheck
```

---

## Architecture Summary

```
               +---------------------+
               |   Client / UI       |
               +----------+----------+
                          |
                          v
               +---------------------+
               |  Gateway Service    |
               |  (FastAPI, port 8000|
               +----------+----------+
                          |
                          v
               +---------------------------+
               |      RAG Service         |
               |       (port 8002)        |
               |---------------------------|
               | 1. Dataset loader         |
               | 2. FAISS semantic search  |
               | 3. Exact-match detector   |
               | 4. Agentic controller     |
               | 5. HF/Gemini fallback     |
               +---------------------------+
```

---

## Why This Is Agentic

A system is “agentic” if it:

1. **Evaluates multiple actions**
2. **Chooses the best one based on reasoning**
3. **Executes a plan**
4. **Self-corrects via fallback behavior**

In this system:

* It chooses between *dataset*, *HF*, *Gemini*, or *Abort*
* It inspects retrieval quality
* It analyzes the question structure (multi-question detection)
* It uses a controller LLM to validate the decision
* It falls back deterministically
* It prevents hallucination

This satisfies the core definition of agentic behavior.

---

## License

MIT
