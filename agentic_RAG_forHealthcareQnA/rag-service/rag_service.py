#!/usr/bin/env python3
"""
Agentic RAG Service for Healthcare Q&A
---------------------------------------
Combines FAISS-based retrieval with LLM generation from HuggingFace and Gemini,
with a lightweight controller that decides which source to use.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Tuple, Optional, Dict
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu

# HuggingFace Hub LLM
from langchain_community.llms import HuggingFaceHub

# Google GenAI (new SDK)
from google import genai

# ------------------ Configuration ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = "/data/healthcare_qa_dataset.jsonl"
LOCAL_DATASET_PATH = os.path.join(BASE_DIR, "healthcare_qa_dataset.jsonl")
DATASET_PATH = DEFAULT_DATASET_PATH if os.path.exists(DEFAULT_DATASET_PATH) else LOCAL_DATASET_PATH

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "rag_index.faiss")
DOCS_PATH = os.path.join(BASE_DIR, "rag_docs.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "rag_embeddings.npy")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.65
DIRECT_ANSWER_THRESHOLD = 0.95
TOP_K = 1

CONTROLLER_TIMEOUT = 6  # seconds to wait for controller decision
LLM_CALL_TIMEOUT = 20   # seconds for LLM answer call

# Model name used for Gemini calls (override with env GEMINI_MODEL)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_REPO_ID = os.getenv("HUGGINGFACEHUB_REPO_ID", "HuggingFaceTB/SmolLM3-3B")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------ Global Variables ------------------
embed_model: Optional[SentenceTransformer] = None
faiss_index: Optional[faiss.Index] = None
documents: List[str] = []
normalized_embeddings: Optional[np.ndarray] = None
hf_llm = None
gemini_client = None  # will hold genai.Client()

# ------------------ Dataset Utilities ------------------
def load_dataset(path: str) -> List[Tuple[str, str]]:
    """Load dataset from JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt = entry.get("prompt")
            completion = entry.get("completion")
            if prompt and completion:
                rows.append((prompt, completion))
    logger.info("Loaded %d QA pairs from dataset", len(rows))
    return rows

def build_documents(rows: List[Tuple[str, str]]) -> List[str]:
    """Build document strings for FAISS index."""
    return [f"Q: {q}\nA: {a}" for q, a in rows]

# ------------------ Embeddings & FAISS ------------------
def ensure_embedding_model() -> None:
    """Load embedding model if not already loaded."""
    global embed_model
    if embed_model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize embeddings for cosine similarity search."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return x / norms

def build_or_load_index(docs: List[str]) -> None:
    """Build or load FAISS index with embeddings."""
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
    logger.info("FAISS index built with %d documents", len(docs))

# ------------------ RAG Search ------------------
def rag_search_sync(query: str, k: int = TOP_K):
    """Synchronous FAISS search."""
    if faiss_index is None:
        raise RuntimeError("FAISS index not initialized")
    ensure_embedding_model()
    q_emb = normalize(embed_model.encode([query], convert_to_numpy=True))
    D, I = faiss_index.search(q_emb.astype("float32"), k)
    results = [(documents[idx], float(dist)) for dist, idx in zip(D[0], I[0]) if 0 <= idx < len(documents)]
    return results

async def async_rag_search(query: str, k: int = TOP_K):
    """Async wrapper for FAISS search."""
    return await asyncio.to_thread(rag_search_sync, query, k)

HEALTHCARE_KEYWORDS = {
    "health", "medicine", "symptom", "diabetes", "blood",
    "cancer", "treatment", "therapy", "disease", "diagnosis"
}

def is_likely_healthcare_question(query: str) -> bool:
    """Heuristic check if question is healthcare-related."""
    return any(kw in query.lower() for kw in HEALTHCARE_KEYWORDS)

def extract_answer_from_doc(doc: str) -> str:
    """Extract answer portion from a retrieved document."""
    if not doc:
        return ""
    if "A:" in doc:
        return doc.split("A:", 1)[-1].strip()
    return doc.strip()

def clean_model_output(text: str) -> str:
    """Clean output from LLMs."""
    if not text:
        return ""
    t = text.strip()
    if "A:" in t and t.count("A:") >= 1:
        t = t.split("A:", 1)[-1].strip()
    if t.startswith("Q:"):
        if "A:" in t:
            t = t.split("A:", 1)[-1].strip()
        else:
            t = t.split("Q:", 1)[-1].strip()
    return " ".join(t.split())

# ------------------ LLM Clients ------------------
def get_hf_llm():
    global hf_llm
    if hf_llm is not None:
        return hf_llm
    if not HUGGINGFACEHUB_API_TOKEN:
        logger.warning("HF token not set; disabling HF LLM")
        return None
    try:
        hf_llm = HuggingFaceHub(
            repo_id=HUGGINGFACEHUB_REPO_ID,
            task="text-generation",
            model_kwargs={"temperature": 0.05, "max_length": 512},
        )
        return hf_llm
    except Exception as e:
        logger.error("Failed to initialize HuggingFaceHub: %s", e)
        return None

def get_gemini_client():
    """
    Initialize and return the genai.Client() object (new Google GenAI SDK).
    The client picks up GEMINI_API_KEY from environment if you don't pass api_key.
    """
    global gemini_client
    if gemini_client is not None:
        return gemini_client

    try:
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            client = genai.Client()  # will read env var GEMINI_API_KEY or GOOGLE_API_KEY
        gemini_client = client
        return gemini_client
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        return None

# ------------------ Controller ------------------
SYSTEM_INSTRUCTION = (
    "You are a healthcare assistant. Answer only healthcare-related questions.\n"
    "When producing an answer, DO NOT repeat the question or the provided context.\n"
    "If you are confident in your answer, return only the final answer text.\n"
    "If the question is not healthcare-related or you cannot answer, reply exactly: \"I don't know.\""
)

CONTROLLER_PROMPT = (
    "You are a controller that decides how to answer a user's healthcare question.\n"
    "Given: 1) the user question, 2) an optional top retrieved doc and its similarity score.\n"
    "Return exactly one of: USE_DATASET, CALL_HF, CALL_GEMINI, or ABORT.\n"
    "Choose USE_DATASET only if the retrieved doc is clearly sufficient.\n"
    "Choose CALL_HF to ask HuggingFace model.\n"
    "Choose CALL_GEMINI to ask Gemini.\n"
    "Choose ABORT to return 'I don't know.'\n"
)

def controller_decision_sync(question: str, retrieved_doc: Optional[str], score: float) -> str:
    """Controller heuristic + LLM decision."""
    prompt = CONTROLLER_PROMPT + "\nUser question:\n" + question
    if retrieved_doc:
        prompt += f"\nTop retrieved doc (score={score:.4f}):\n{retrieved_doc}\n"

    client = get_hf_llm() or get_gemini_client()
    if client is None:
        # fallback heuristic
        if score >= DIRECT_ANSWER_THRESHOLD:
            return "USE_DATASET"
        if score >= SIMILARITY_THRESHOLD:
            return "CALL_HF"
        if is_likely_healthcare_question(question):
            return "CALL_HF"
        return "ABORT"

    try:
        # If client is a HuggingFaceHub object, call it with prompt as before
        if isinstance(client, HuggingFaceHub):
            res = client(prompt)
            text = res.get("text") if isinstance(res, dict) else str(res)
        else:
            # It's genai.Client: use client.models.generate_content
            # We ask the controller LLM to return exactly one token.
            res = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            # response has .text
            text = getattr(res, "text", "") or ""
        text = clean_model_output(text).upper()
        for token in ("USE_DATASET", "CALL_HF", "CALL_GEMINI", "ABORT"):
            if token in text:
                return token
    except Exception as e:
        logger.warning("Controller LLM failed: %s", e)

    # fallback heuristic
    if score >= DIRECT_ANSWER_THRESHOLD:
        return "USE_DATASET"
    if score >= SIMILARITY_THRESHOLD:
        return "CALL_HF"
    if is_likely_healthcare_question(question):
        return "CALL_HF"
    return "ABORT"

async def async_controller_decision(question: str, retrieved_doc: Optional[str], score: float) -> str:
    return await asyncio.to_thread(controller_decision_sync, question, retrieved_doc, score)

# ------------------ LLM Calls ------------------
def call_llm_with_context_sync(query: str, context: Optional[str] = None, prefer: str = "HF") -> Optional[str]:
    prompt = SYSTEM_INSTRUCTION + "\n\n" + (f"Context:\n{context}\n\n" if context else "") + f"Question:\n{query}\nAnswer:"

    # determine primary and fallback clients
    if prefer == "HF":
        primary = get_hf_llm()
        fallback = get_gemini_client()
    else:
        primary = get_gemini_client()
        fallback = get_hf_llm()

    # Helper to call client generically
    def call_client(client_obj, prompt_text):
        if client_obj is None:
            return None
        try:
            if isinstance(client_obj, HuggingFaceHub):
                res = client_obj(prompt_text)
                text = res.get("text") if isinstance(res, dict) else str(res)
                return text
            else:
                # genai.Client usage
                res = client_obj.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt_text,
                )
                # res.text holds textual content
                return getattr(res, "text", None)
        except Exception as e:
            logger.warning("LLM client call failed: %s", e)
            return None

    # Try primary then fallback
    text = call_client(primary, prompt)
    if text:
        return clean_model_output(text)
    text = call_client(fallback, prompt)
    if text:
        return clean_model_output(text)
    return None

async def async_call_llm_with_context(query: str, context: Optional[str] = None, prefer: str = "HF") -> Optional[str]:
    # Note: the genai SDK also provides an async client under client.aio.*.
    # For simplicity and compatibility we run the sync call in a thread.
    return await asyncio.to_thread(call_llm_with_context_sync, query, context, prefer)

# ------------------ Agentic Loop ------------------
async def agentic_answer(query: str) -> Tuple[str, Optional[str], float]:
    """Agentic decision loop: retrieval -> controller -> LLM -> fallback."""
    try:
        results = await async_rag_search(query, k=TOP_K)
    except Exception as e:
        logger.warning("Retrieval failed: %s", e)
        results = []

    top_doc, top_score = (results[0] if results else (None, -1.0))

    # High confidence dataset answer
    if top_doc and top_score >= DIRECT_ANSWER_THRESHOLD:
        return extract_answer_from_doc(top_doc), "dataset (high-confidence)", top_score

    decision = await async_controller_decision(query, top_doc, top_score)
    logger.info("Controller decision: %s", decision)

    # Decision execution with fallbacks
    async def try_llm(prefer: str):
        try:
            return await asyncio.wait_for(async_call_llm_with_context(query, top_doc, prefer=prefer), timeout=LLM_CALL_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("%s LLM timed out", prefer)
            return None
        except Exception as e:
            logger.warning("%s LLM call error: %s", prefer, e)
            return None

    if decision == "USE_DATASET" and top_doc:
        return extract_answer_from_doc(top_doc), "dataset", top_score

    if decision in ("CALL_HF", "CALL_GEMINI"):
        preferred = "HF" if decision == "CALL_HF" else "GEMINI"
        ans = await try_llm(preferred)
        if not ans or ans.strip().lower() == "i don't know.":
            # fallback
            fallback = "GEMINI" if preferred == "HF" else "HF"
            ans = await try_llm(fallback)
        if ans and ans.strip().lower() != "i don't know.":
            return ans, f"LLM({preferred})", top_score
        if top_doc:
            return extract_answer_from_doc(top_doc), "dataset (fallback)", top_score
        return "I don't know.", None, top_score

    if decision == "ABORT":
        return "I don't know.", None, top_score

    # Default fallback HF -> Gemini -> dataset
    for prefer in ["HF", "GEMINI"]:
        ans = await try_llm(prefer)
        if ans and ans.strip().lower() != "i don't know.":
            return ans, f"LLM({prefer})", top_score
    if top_doc:
        return extract_answer_from_doc(top_doc), "dataset (fallback)", top_score
    return "I don't know.", None, top_score

# ------------------ Initialization ------------------
def initialize() -> None:
    """Initialize FAISS, embeddings, and LLMs."""
    try:
        rows = load_dataset(DATASET_PATH)
        docs = build_documents(rows)
        build_or_load_index(docs)
        # Initialize LLM clients (they will log warnings if keys missing)
        get_hf_llm()
        get_gemini_client()
        logger.info("RAG system initialized with %d docs", len(docs))
    except Exception as e:
        logger.error("Initialization failed: %s", e)
        raise

# ------------------ FastAPI ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        initialize()
    except Exception:
        logger.exception("Initialization during FastAPI startup failed")
    yield

app = FastAPI(title="Agentic RAG Service (Healthcare)", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: Optional[str] = None
    score: Optional[float] = None

@app.get("/")
async def root():
    return {"status": "ok", "docs": len(documents)}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        answer, source, score = await agentic_answer(query)
        return QueryResponse(answer=answer, source=source, score=score)
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

# ---------- Evaluation Functions ----------
def evaluate_retrieval(rows: List[Tuple[str, str]], docs: List[str]) -> Dict[str, float]:
    hit_at_1, mrr_sum = 0, 0.0
    total = len(rows)

    for i, (q, gold_a) in enumerate(rows):
        results = rag_search_sync(q, k=5)

        # Hit Rate (is the correct doc in the top k?)
        is_hit = False
        for retrieved_doc, _ in results:
            if gold_a.lower() in retrieved_doc.lower():
                is_hit = True
                break
        if is_hit:
            hit_at_1 += 1

        # MRR (what's the rank of the first correct doc?)
        reciprocal_rank = 0.0
        for rank, (retrieved_doc, _) in enumerate(results):
            if gold_a.lower() in retrieved_doc.lower():
                reciprocal_rank = 1.0 / (rank + 1)
                break
        mrr_sum += reciprocal_rank

    hit_rate = hit_at_1 / total if total > 0 else 0
    mrr = mrr_sum / total if total > 0 else 0
    return {"hit_rate": hit_rate, "mrr": mrr}


def evaluate_generation(rows: List[Tuple[str, str]]) -> Dict[str, float]:
    total, correct_match, bleu_scores = 0, 0, []
    for q, gold_a in rows:
        results = rag_search_sync(q, k=TOP_K)
        top_doc, score = results[0] if results else ("", 0.0)

        pred_answer = ""
        if top_doc and score >= SIMILARITY_THRESHOLD:
            llm_answer = call_llm_with_context_sync(q, top_doc)
            pred_answer = llm_answer if llm_answer else extract_answer_from_doc(top_doc)
        elif is_likely_healthcare_question(q):
            llm_answer = call_llm_with_context_sync(q)
            pred_answer = llm_answer if llm_answer else "I don't know."
        else:
            pred_answer = "I can only answer questions related to healthcare. Please ask a healthcare-related question."
        
        bleu = sentence_bleu([gold_a.split()], pred_answer.split())
        bleu_scores.append(bleu)
        
        if gold_a.lower() in pred_answer.lower():
            correct_match += 1
        
        total += 1
    
    precision_at_1 = correct_match / total if total > 0 else 0
    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    
    return {"precision_at_1": precision_at_1, "avg_bleu": avg_bleu}

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    if "--eval" in sys.argv:
        initialize()
        rows = load_dataset(DATASET_PATH)
        docs = build_documents(rows)
        
        # We'll use a small subset for quick evaluation to avoid long runtime
        eval_rows = rows[:50]
        
        print(f"Running evaluation on {len(eval_rows)} samples...")
        
        retrieval_metrics = evaluate_retrieval(eval_rows, docs)
        print("\n--- Retrieval Metrics (Evaluates search quality) ---")
        print(f"Hit Rate @5: {retrieval_metrics['hit_rate']:.2f} (Percentage of queries where the correct doc was in the top 5 results)")
        print(f"Mean Reciprocal Rank (MRR) @5: {retrieval_metrics['mrr']:.2f} (Average rank of the first correct doc)")

        generation_metrics = evaluate_generation(eval_rows)
        print("\n--- Generation Metrics (Evaluates answer quality) ---")
        print(f"Precision@1 (Strict Match): {generation_metrics['precision_at_1']:.2f} (Percentage of exact or near-exact answers)")
        print(f"Avg BLEU: {generation_metrics['avg_bleu']:.2f} (Similarity to the reference answer)")

    else:
        uvicorn.run("rag_service:app", host="0.0.0.0", port=8002, reload=True)
