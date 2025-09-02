"""
Improved `rag_service.py` for the Agentic RAG healthcare demo (lifespan version).

Key improvements:
- Uses FastAPI lifespan context manager instead of deprecated @app.on_event
- No hardcoded secrets: loads HUGGINGFACEHUB_API_TOKEN from env or .env
- Persists FAISS index, documents, and embeddings for reproducibility
- Uses cosine similarity (normalized embeddings + IndexFlatIP)
- Deterministic router: RAG → LLM-with-context → LLM-without-context → refusal
- Fallback if HF token missing: return stored answers for in-scope queries
- Clear logging and error handling

Usage:
  - Create `.env` file in repo root with:
      HUGGINGFACEHUB_API_TOKEN=hf_xxx
      HUGGINGFACEHUB_REPO_ID=HuggingFaceH4/zephyr-7b-beta  # optional override
  - Ensure `healthcare_qa_dataset.jsonl` exists next to this file.
  - Run: `python rag_service.py`

"""

import os

ATLAS_URI = os.getenv("ATLAS_CONNECTION_STRING")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

import json
import logging
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from langchain_community.llms import HuggingFaceHub

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "healthcare_qa_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "rag_index.faiss")
DOCS_PATH = os.path.join(BASE_DIR, "rag_docs.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "rag_embeddings.npy")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.65
TOP_K = 1

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_REPO_ID = os.getenv("HUGGINGFACEHUB_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------- Globals ----------
embed_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.Index] = None
documents: List[str] = []
normalized_embeddings: Optional[np.ndarray] = None
llm = None

# ---------- Utilities ----------
def load_dataset(path: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry.get("prompt")
            completion = entry.get("completion")
            if prompt and completion:
                rows.append((prompt, completion))
    return rows


def build_documents(rows: List[Tuple[str, str]]) -> List[str]:
    return [f"Q: {q}\nA: {a}" for q, a in rows]


def ensure_embedding_model():
    global embed_model
    if embed_model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return x / norms


def build_or_load_index(docs: List[str]):
    global faiss_index, documents, normalized_embeddings
    ensure_embedding_model()

    if all(os.path.exists(p) for p in [FAISS_INDEX_PATH, DOCS_PATH, EMBEDDINGS_PATH]):
        try:
            logger.info("Loading FAISS index and docs from disk")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)
            normalized_embeddings = np.load(EMBEDDINGS_PATH)
            if len(documents) != normalized_embeddings.shape[0]:
                raise RuntimeError("Docs/embeddings mismatch")
            return
        except Exception as e:
            logger.warning("Failed loading index: %s. Rebuilding...", e)

    logger.info("Building new FAISS index...")
    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    normalized_embeddings = normalize(embeddings)
    dim = normalized_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(normalized_embeddings.astype("float32"))

    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, normalized_embeddings)
    documents = docs


def rag_search(query: str, k: int = TOP_K):
    if faiss_index is None:
        raise RuntimeError("FAISS index not initialized")
    ensure_embedding_model()
    q_emb = normalize(embed_model.encode([query], convert_to_numpy=True))
    D, I = faiss_index.search(q_emb.astype("float32"), k)
    results = [(documents[idx], float(dist)) for dist, idx in zip(D[0], I[0]) if 0 <= idx < len(documents)]
    return results


HEALTHCARE_KEYWORDS = {"health", "medicine", "symptom", "diabetes", "blood", "cancer", "treatment", "therapy"}

def is_likely_healthcare_question(query: str) -> bool:
    return any(kw in query.lower() for kw in HEALTHCARE_KEYWORDS)


# ---------- LLM ----------
def get_llm():
    global llm
    if llm is not None:
        return llm
    if not HUGGINGFACEHUB_API_TOKEN:
        logger.warning("HF token not set; disabling LLM")
        return None
    llm = HuggingFaceHub(repo_id=HUGGINGFACEHUB_REPO_ID, model_kwargs={"temperature": 0.05, "max_length": 512})
    return llm


SYSTEM_INSTRUCTION = (
    "You are a healthcare assistant. Answer only healthcare-related questions.\n"
    "If not healthcare, reply: 'I can only answer questions related to healthcare. Please ask a healthcare-related question.'"
)

def call_llm_with_context(query: str, context: Optional[str] = None) -> str:
    client = get_llm()
    if client is None:
        raise RuntimeError("LLM unavailable")
    prompt = f"{SYSTEM_INSTRUCTION}\n\n" + (f"Context:\n{context}\n\n" if context else "") + f"Question:\n{query}\nAnswer:"
    res = client(prompt)
    return res["text"].strip() if isinstance(res, dict) and "text" in res else str(res).strip()


# ---------- Lifespan ----------
@asynccontextmanager
def lifespan(app: FastAPI):
    try:
        rows = load_dataset(DATASET_PATH)
        docs = build_documents(rows)
        build_or_load_index(docs)
        get_llm()
        logger.info("RAG system initialized with %d docs", len(docs))
    except Exception as e:
        logger.error("Initialization failed: %s", e)
    yield


# ---------- FastAPI ----------
app = FastAPI(title="Agentic RAG Service (Healthcare)", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: Optional[str] = None
    score: Optional[float] = None


@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        results = rag_search(query, k=TOP_K)
        top_doc, top_score = results[0] if results else (None, -1.0)

        if top_doc and top_score >= SIMILARITY_THRESHOLD:
            if llm:
                try:
                    ans = call_llm_with_context(query, top_doc)
                    return QueryResponse(answer=ans, source=top_doc, score=top_score)
                except Exception:
                    logger.exception("LLM failed; returning stored answer")
                    return QueryResponse(answer=top_doc, source=top_doc, score=top_score)
            return QueryResponse(answer=top_doc, source=top_doc, score=top_score)

        if is_likely_healthcare_question(query):
            if llm:
                ans = call_llm_with_context(query)
                return QueryResponse(answer=ans, source=None, score=top_score)
            return QueryResponse(answer="I can only answer healthcare questions covered by my dataset.")

        return QueryResponse(answer="I can only answer questions related to healthcare. Please ask a healthcare-related question.")
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("rag_service:app", host="0.0.0.0", port=8002, reload=True)
