#!/usr/bin/env python3
import os, sys, json, logging, asyncio, re
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_distances

from langchain_community.llms import HuggingFaceHub
from google import genai

# -------------------- Config --------------------
load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "/data")
DATASET_PATH = os.path.join(DATA_DIR, "healthcare_qa_dataset.jsonl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "rag_index.faiss")
DOCS_PATH = os.path.join(DATA_DIR, "rag_docs.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "rag_embeddings.npy")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.70
DIRECT_ANSWER_THRESHOLD = 0.80
TOP_K = 1
LLM_CALL_TIMEOUT = 20

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_REPO = os.getenv("HUGGINGFACEHUB_REPO_ID", "HuggingFaceTB/SmolLM3-3B")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("rag")

# -------------------- Globals --------------------
embed_model = None
faiss_index = None
documents: List[str] = []
normalized_embeddings = None
hf_llm = None
gemini_client = None

# -------------------- Dataset & Index --------------------
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if "prompt" in e and "completion" in e:
                    rows.append((e["prompt"], e["completion"]))
    logger.info("Loaded %d QA pairs", len(rows))
    return rows

def build_documents(rows):
    return [f"Q: {q}\nA: {a}" for q, a in rows]

def ensure_embed():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1e-9
    return x / n

def build_or_load_index(docs):
    global faiss_index, documents, normalized_embeddings
    ensure_embed()
    if all(os.path.exists(p) for p in [FAISS_INDEX_PATH, DOCS_PATH, EMBEDDINGS_PATH]):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            documents = json.load(open(DOCS_PATH, "r", encoding="utf-8"))
            normalized_embeddings = np.load(EMBEDDINGS_PATH)
            logger.info("Loaded FAISS index (%d docs)", len(documents))
            return
        except Exception as e:
            logger.warning("Failed loading index (%s). Rebuilding...", e)

    emb = embed_model.encode(docs, convert_to_numpy=True)
    normalized_embeddings = normalize(emb)
    idx = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    idx.add(normalized_embeddings.astype("float32"))
    faiss_index = idx
    documents = docs

    faiss.write_index(idx, FAISS_INDEX_PATH)
    json.dump(docs, open(DOCS_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    np.save(EMBEDDINGS_PATH, normalized_embeddings)
    logger.info("Built FAISS index (%d docs)", len(docs))

# -------------------- Retrieval --------------------
def rag_search_sync(q, k=TOP_K):
    ensure_embed()
    q_emb = normalize(embed_model.encode([q], convert_to_numpy=True))
    D, I = faiss_index.search(q_emb.astype("float32"), k)
    return [(documents[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(documents)]

async def async_rag_search(q, k=TOP_K):
    return await asyncio.to_thread(rag_search_sync, q, k)

# -------------------- Multi-Question Detection --------------------
def split_units(q):
    parts = re.split(r"[.?;]+", q.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts

    soft = re.split(r'\b(and|or|also|vs|while)\b', q, flags=re.I)
    merged, buf = [], ""
    for seg in soft:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg.split()) < 3:
            buf += " " + seg
        else:
            merged.append((buf + " " + seg).strip() if buf else seg)
            buf = ""
    if buf:
        merged.append(buf)
    return merged if len(merged) >= 2 else [q]

def semantic_is_multi(q):
    ensure_embed()
    units = split_units(q)
    if len(units) <= 1:
        logger.info("[DEBUG] semantic 1-unit")
        return False

    emb = embed_model.encode(units, convert_to_numpy=True)
    if emb.shape[0] <= 1:
        return False

    dist = cosine_distances(emb)
    mx, avg = float(np.max(dist)), float(np.mean(dist))
    logger.info(f"[DEBUG] units={len(units)} max={mx:.3f} avg={avg:.3f}")

    if mx > 0.40 or avg > 0.28:
        logger.info("[DEBUG] semantic_multi=True")
        return True
    return False

# -------------------- Heuristics --------------------
HEALTHCARE_KEYWORDS = {
    "health","medicine","symptom","diabetes","cancer",
    "blood","treatment","therapy","disease","diagnosis"
}

def is_health(q): return any(w in q.lower() for w in HEALTHCARE_KEYWORDS)

def tokenize(s): return re.findall(r"\w+", s.lower())

def overlap_fraction(q, doc):
    if not doc: return 0.0
    q_t = set(tokenize(q))
    d_t = set(tokenize(doc))
    return len(q_t & d_t) / len(q_t) if q_t else 0.0

def extract_answer(doc): return doc.split("A:", 1)[-1].strip()

def clean_output(t):
    if not t: return ""
    t = t.strip()
    if "A:" in t: t = t.split("A:",1)[-1]
    if t.startswith("Q:"): t = t.split("Q:",1)[-1]
    return " ".join(t.split())

# -------------------- LLM Clients --------------------
def get_hf_llm():
    global hf_llm
    if hf_llm: return hf_llm
    if not HF_TOKEN: return None
    try:
        hf_llm = HuggingFaceHub(
            repo_id=HF_REPO,
            task="text-generation",
            model_kwargs={"temperature":0.05,"max_length":512},
        )
        return hf_llm
    except Exception as e:
        logger.error("HF init error: %s", e)
        return None

def get_gemini_client():
    global gemini_client
    if gemini_client: return gemini_client
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
        return gemini_client
    except Exception as e:
        logger.error("Gemini init error: %s", e)
        return None

# -------------------- Controller --------------------
SYS_INSTRUCT = (
    "You are a healthcare assistant. Answer only healthcare questions. "
    "If unsure, reply: I don't know."
)
CTRL_PROMPT = "Decide answer source: USE_DATASET, CALL_HF, CALL_GEMINI, ABORT.\n"

# -------------------- Controller --------------------

def controller_decision_sync(q, doc, score):
    """
    Decide the answer source using a strict priority hierarchy:

        1. Strong retrieval match  → USE_DATASET
        2. Multi-question          → CALL_HF (deterministic)
        3. Very low overlap        → CALL_HF
        4. Controller LLM advisory (HF or Gemini)
        5. Heuristic fallback based only on similarity + domain
    """

    # ---------- Base stats ----------
    multi = semantic_is_multi(q)
    frac  = overlap_fraction(q, doc) if doc else -1.0
    logger.info(f"[DEBUG] controller multi={multi} frac={frac:.3f} score={score:.3f}")

    # ---------- RULE 1: Strong dataset match ----------
    if doc and score >= DIRECT_ANSWER_THRESHOLD and frac >= 0.25:
        logger.info("[DEBUG] strong retrieval → USE_DATASET")
        return "USE_DATASET"

    # ---------- RULE 2: Multi-question always uses HF ----------
    if multi:
        return "CALL_HF"

    # ---------- RULE 3: Low lexical overlap → LLM ----------
    if doc and frac < 0.18:
        return "CALL_HF"

    # ---------- Build controller prompt ----------
    prompt = CTRL_PROMPT + f"User:\n{q}"
    if doc:
        snippet = (doc[:1000] + "...") if len(doc) > 1000 else doc
        prompt += f"\nDoc(score={score:.3f}):\n{snippet}"

    # ---------- Choose controller LLM ----------
    client = get_hf_llm() or get_gemini_client()

    # ---------- RULE 4: No LLM available → heuristic fallback ----------
    if client is None:
        if score >= DIRECT_ANSWER_THRESHOLD:
            return "USE_DATASET"
        if score >= SIMILARITY_THRESHOLD:
            return "CALL_HF"
        if is_health(q):
            return "CALL_HF"
        return "ABORT"

    # ---------- Ask controller LLM ----------
    try:
        if isinstance(client, HuggingFaceHub):
            out = client(prompt)
            raw = out.get("text") if isinstance(out, dict) else str(out)
        else:
            res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = getattr(res, "text", "") or ""

        logger.debug("[DEBUG] controller LLM raw output: %s", raw[:300])

        t = clean_output(raw).upper()

        # Try to read explicit token from LLM
        for tok in ("USE_DATASET", "CALL_HF", "CALL_GEMINI", "ABORT"):
            if tok in t:
                logger.info("[DEBUG] controller LLM decided %s", tok)
                return tok

    except Exception as e:
        logger.warning("Controller LLM error: %s", e)

    # ---------- RULE 5: Heuristic fallback ----------
    if score >= DIRECT_ANSWER_THRESHOLD:
        return "USE_DATASET"
    if score >= SIMILARITY_THRESHOLD:
        return "CALL_HF"
    if is_health(q):
        return "CALL_HF"

    return "ABORT"

async def async_controller_decision(q, doc, score):
    """Async wrapper so FastAPI does not break."""
    return await asyncio.to_thread(controller_decision_sync, q, doc, score)

# -------------------- LLM calls --------------------
def call_llm_sync(q, ctx, pref):
    prompt = SYS_INSTRUCT + "\n\n"
    if ctx:
        prompt += f"Context:\n{ctx}\n\n"
    prompt += f"Question:\n{q}\nAnswer:"

    primary  = get_hf_llm() if pref == "HF" else get_gemini_client()
    fallback = get_gemini_client() if pref == "HF" else get_hf_llm()

    def call(c):
        if not c:
            return None
        try:
            if isinstance(c, HuggingFaceHub):
                out = c(prompt)
                return out.get("text") if isinstance(out, dict) else str(out)
            res = c.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return getattr(res, "text", None)
        except Exception:
            return None

    t = call(primary)
    if t:
        return clean_output(t)

    t = call(fallback)
    if t:
        return clean_output(t)

    return None


async def async_call_llm(q, ctx, pref):
    return await asyncio.to_thread(call_llm_sync, q, ctx, pref)

# -------------------- Agentic RAG --------------------
def normalize_question(q: str):
    q = q.lower().strip()
    q = re.sub(r'[^a-z0-9\s]', '', q)   # remove punctuation
    q = re.sub(r'\s+', ' ', q)          # collapse spaces
    return q


async def agentic_answer(q):
    # Step 1: Retrieve candidates
    try:
        results = await async_rag_search(q)
    except Exception:
        results = []

    doc, score = results[0] if results else (None, -1.0)

    # Step 2: STRICT exact-match check (normalized)
    if doc:
        q_norm = normalize_question(q)
        doc_q_raw = doc.split("A:", 1)[0].replace("Q:", "")
        doc_norm = normalize_question(doc_q_raw)

        if q_norm == doc_norm:
            logger.info("[DEBUG] exact-match detected → using dataset answer")
            return extract_answer(doc), "dataset (exact-match)", 1.0

    # Step 3: Multi-question and overlap check
    multi = semantic_is_multi(q)
    frac = overlap_fraction(q, doc) if doc else -1.0
    logger.info(f"[DEBUG] agentic multi={multi} frac={frac:.3f} score={score:.3f}")

    # Step 4: High-confidence direct dataset use
    if doc and score >= DIRECT_ANSWER_THRESHOLD and frac >= 0.25:
        return extract_answer(doc), "dataset (high-confidence)", score

    # Step 5: Controller decision
    decision = await async_controller_decision(q, doc, score)
    logger.info(f"Controller decision: {decision}")

    async def try_llm(pref):
        try:
            return await asyncio.wait_for(
                async_call_llm(q, doc, pref),
                timeout=LLM_CALL_TIMEOUT
            )
        except Exception:
            return None

    # Step 6: USE_DATASET explicitly chosen
    if decision == "USE_DATASET" and doc:
        return extract_answer(doc), "dataset", score

    # Step 7: LLM path
    if decision in ("CALL_HF", "CALL_GEMINI"):
        pref = "HF" if decision == "CALL_HF" else "GEMINI"

        # primary
        ans = await try_llm(pref)
        if ans and ans.lower() != "i don't know.":
            return ans, f"LLM({pref})", score

        # fallback
        alt = "GEMINI" if pref == "HF" else "HF"
        ans = await try_llm(alt)
        if ans:
            return ans, f"LLM({alt})", score

        # dataset fallback
        if doc:
            return extract_answer(doc), "dataset (fallback)", score

        return "I don't know.", None, score

    # Step 8: Abort fallback
    if decision == "ABORT":
        if doc:
            return extract_answer(doc), "dataset (fallback)", score
        return "I don't know.", None, score

    # Step 9: Final safety
    for pref in ("HF", "GEMINI"):
        ans = await try_llm(pref)
        if ans:
            return ans, f"LLM({pref})", score

    if doc:
        return extract_answer(doc), "dataset (fallback)", score

    return "I don't know.", None, score

# -------------------- Init & API --------------------
def initialize():
    rows = load_dataset(DATASET_PATH)
    docs = build_documents(rows)
    build_or_load_index(docs)
    get_hf_llm()
    get_gemini_client()
    logger.info("RAG initialized (%d docs)", len(docs))

@asynccontextmanager
async def lifespan(app):
    initialize()
    yield

app = FastAPI(title="Agentic RAG Service", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source: Optional[str]
    score: Optional[float]

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    if faiss_index is None:
        raise HTTPException(503,"RAG not initialized")
    q = req.query.strip()
    if not q:
        raise HTTPException(400,"Empty query")
    ans, src, score = await agentic_answer(q)
    return QueryResponse(answer=ans, source=src, score=score)

@app.get("/debug")
async def debug_env():
    """Show which important env vars are actually loaded inside the container."""
    return {
        "DATA_DIR": os.getenv("DATA_DIR"),
        "GEMINI_API_KEY_set": bool(os.getenv("GEMINI_API_KEY")),
        "HF_TOKEN_set": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")),
        "HF_REPO": os.getenv("HUGGINGFACEHUB_REPO_ID"),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
    }

@app.get("/healthcheck")
async def healthcheck():
    results = {}

    # --- Check Gemini ---
    try:
        client = get_gemini_client()
        if client:
            res = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="ping"
            )
            results["gemini"] = "ok" if getattr(res, "text", None) else "no_text"
        else:
            results["gemini"] = "not_initialized"
    except Exception as e:
        results["gemini"] = f"error: {str(e)}"


    # --- Check HuggingFaceHub ---
    try:
        hf = get_hf_llm()
        if hf:
            out = hf("ping")
            if isinstance(out, dict):
                txt = out.get("text", None)
            else:
                txt = str(out)
            results["huggingface"] = "ok" if txt else "no_text"
        else:
            results["huggingface"] = "not_initialized"
    except Exception as e:
        results["huggingface"] = f"error: {str(e)}"

    # --- Check FAISS ---
    results["faiss_loaded"] = faiss_index is not None
    results["documents_count"] = len(documents)

    return results


# -------------------- Evaluation --------------------
def evaluate_retrieval(rows):
    hit, mrr = 0, 0.0
    for q, gold in rows:
        res = rag_search_sync(q, 5)
        docs = [d for d,_ in res]
        if any(gold.lower() in d.lower() for d in docs): hit += 1
        rr = 0.0
        for i,(d,_) in enumerate(res):
            if gold.lower() in d.lower():
                rr = 1/(i+1)
                break
        mrr += rr
    return {"hit_rate": hit/len(rows), "mrr": mrr/len(rows)}

def evaluate_generation(rows):
    tot, corr, bleus = 0, 0, []
    for q, gold in rows:
        res = rag_search_sync(q)
        doc, score = res[0] if res else ("", 0)
        if doc and score >= SIMILARITY_THRESHOLD:
            ans = call_llm_sync(q, doc, "HF")
        elif is_health(q):
            ans = call_llm_sync(q, None, "HF")
        else:
            ans = "I don't know."
        bleus.append(sentence_bleu([gold.split()], ans.split()))
        if gold.lower() in ans.lower(): corr += 1
        tot += 1
    return {
        "precision_at_1": corr/tot,
        "avg_bleu": float(np.mean(bleus))
    }

# -------------------- Main --------------------
if __name__ == "__main__":
    if "--eval" in sys.argv:
        initialize()
        rows = load_dataset(DATASET_PATH)
        print("Retrieval:", evaluate_retrieval(rows[:50]))
        print("Generation:", evaluate_generation(rows[:50]))
    else:
        uvicorn.run("rag_service:app", host="0.0.0.0", port=8002, reload=True)
