# app/demo.py — self-contained Streamlit demo (HF-only encoder), with Fusion + Hybrid
from __future__ import annotations
import json, re, hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import jsonlines
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

import os, google.generativeai as genai
from google.generativeai import GenerativeModel
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY",""))

# UI toggle
USE_GEMINI_HYDE = st.checkbox("Use Gemini for hypos", value=True)

# Cache the model in the app
@st.cache_resource(show_spinner=False)
def get_gemini_model():
    return GenerativeModel("gemini-2.5-flash")  # or "gemini-2.5-pro"

_gemini = get_gemini_model()

PROMPT = (
    "Write a short, factual paragraph (2–4 sentences) in Wikipedia style that could answer the question. "
    "Use real entity names and concrete facts. Include the specific city if relevant. "
    "Do NOT restate the question. Do NOT invent facts.\n\n"
    "Question: {q}\nPassage:"
)

# -----------------------------
# Paths & basic setup
# -----------------------------
root = Path(__file__).resolve().parents[1]
meta = json.load(open(root / "runs" / "meta.json", "r", encoding="utf-8"))

st.set_page_config(page_title="HyDE Demo", layout="wide")
st.title("HyDE (ACL'23) — Minimal Demo")

mode = st.radio(
    "Mode",
    [
        "Baseline",
        "HyDE (1 hypo)",
        "Multi-HyDE (2 hypos)",
        "Fusion (Baseline+Multi-HyDE)",
        "Hybrid (BM25+Baseline+Multi-HyDE)",
    ],
    horizontal=True,
)
k = st.slider("Top-k", 5, 30, 10)
q = st.text_input("Question", "Who wrote the Federalist Papers and what was the purpose?")

# -----------------------------
# Config: keep CPU-only; avoid .to(...)
# -----------------------------
ENCODER_MODEL = "intfloat/e5-base-v2"
HYDE_MODEL    = "google/flan-t5-small"   # fast on CPU
HYPO_MAX_NEW  = 80
HYPO_TEMP     = 0.1
HYPO_TOP_P    = 0.9

# -----------------------------
# Inline index wrapper/loader
# -----------------------------
class _Index:
    def __init__(self, faiss_index=None, sklearn_nn=None):
        self._faiss = faiss_index
        self._sk = sklearn_nn
    def search(self, qvecs: np.ndarray, k: int = 10):
        if self._faiss is not None:
            D, I = self._faiss.search(qvecs.astype("float32"), k)
            return I, D
        D, I = self._sk.kneighbors(qvecs, n_neighbors=k, return_distance=True)
        return I, D

def load_index_inline(path_faiss: Path, path_sklearn: Path) -> _Index:
    if path_faiss.exists():
        import faiss  # type: ignore
        return _Index(faiss_index=faiss.read_index(str(path_faiss)))
    else:
        import joblib
        return _Index(sklearn_nn=joblib.load(str(path_sklearn)))

index = load_index_inline(root / "runs" / "index.faiss", root / "runs" / "index.sklearn")

# Load corpus passages so we can display evidence snippets
passages = list(jsonlines.open(str(root / "data" / "passages.jsonl")))

def highlight_snippet(text: str, query: str, max_chars: int = 400) -> str:
    # naive highlight of query tokens (length > 2)
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    s = text
    for t in terms:
        s = re.sub(fr"(?i)\b{re.escape(t)}\b", r"**\g<0>**", s)
    return (s[:max_chars] + "…") if len(s) > max_chars else s

# -----------------------------
# E5 encoder via plain HF (no SentenceTransformer)
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_e5():
    tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", use_fast=True)
    mdl = AutoModel.from_pretrained(
        "intfloat/e5-base-v2",
        low_cpu_mem_usage=False,   # <-- force full weights init (no meta)
        device_map=None            # <-- ensure CPU map
    )
    mdl.eval()
    return tok, mdl

tok_e5, mdl_e5 = get_e5()

@torch.inference_mode()
def e5_encode(texts: list[str]) -> np.ndarray:
    """Mean-pool last hidden state, L2-normalize."""
    batch = tok_e5(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")  # CPU tensors
    out = mdl_e5(**batch)
    emb = out.last_hidden_state.mean(dim=1)                    # [B, H]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

# -----------------------------
# HyDE generator (Flan-T5 small) — keep on CPU
# -----------------------------
tok_hyde = AutoTokenizer.from_pretrained(HYDE_MODEL)
mdl_hyde = AutoModelForSeq2SeqLM.from_pretrained(HYDE_MODEL)  # <-- stays on CPU; no .to()
mdl_hyde.eval()


def gen_hypo(question: str) -> str:
    if USE_GEMINI_HYDE:
        try:
            resp = _gemini.generate_content(
                PROMPT.format(q=question),
                generation_config={"max_output_tokens": 80, "temperature": 0.1, "top_p": 0.9},
            )
            return (resp.text or "").strip()
        except Exception as e:
            st.warning(f"Gemini call failed; falling back to local model. ({e})")

    # FALLBACK: your existing local flan-t5-small path:
    x = tok_hyde(PROMPT.format(q=question), return_tensors="pt")
    y = mdl_hyde.generate(**x, max_new_tokens=80, do_sample=False)
    return tok_hyde.decode(y[0], skip_special_tokens=True).strip()

# -----------------------------
# Simple in-memory cache (per session) for hypos
# -----------------------------
if "hyde_cache" not in st.session_state:
    st.session_state["hyde_cache"] = {}

def cache_key(question: str, n: int) -> str:
    h = hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:10]
    return f"{h}::n={n}"

def get_hypos(question: str, n: int):
    key = cache_key(question, n)
    cache = st.session_state["hyde_cache"]
    if key not in cache:
        hypos = [gen_hypo(question) for _ in range(n)]
        cache[key] = hypos
    return cache[key]

# -----------------------------
# BM25 over passages for Hybrid
# -----------------------------
pass_texts = [rec["text"] for rec in jsonlines.open(str(root / "data" / "passages.jsonl"))]
def tok_simple(s: str): return re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok_simple(t) for t in pass_texts])

def bm25_search(question: str, k: int = 20):
    scores = bm25.get_scores(tok_simple(question))
    idx = np.argsort(-np.array(scores))[:k]
    return idx.tolist()

# -----------------------------
# Fusion helper
# -----------------------------
def rrf_fuse_many(lists, k=10, K=60):
    scores = {}
    for lst in lists:
        for r, pid in enumerate(lst, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (K + r)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]

# -----------------------------
# Search functions
# -----------------------------
def search_baseline(question: str, k: int):
    qv = e5_encode(["query: " + question])
    I, D = index.search(qv.astype("float32"), k)
    return I[0], D[0]

def search_hyde(question: str, k: int, n: int = 1, show_hypos: bool = True):
    hypos = get_hypos(question, n)
    if show_hypos:
        with st.expander("Show hypothetical passages", expanded=False):
            for i, h in enumerate(hypos, 1):
                st.code(f"[{i}] {h}")
    hv = e5_encode(["query: " + h for h in hypos]).mean(axis=0)  # avg Multi-HyDE
    I, D = index.search(hv[None, :].astype("float32"), k)
    return I[0], D[0]

# -----------------------------
# UI action
# -----------------------------
if st.button("Search"):
    with st.spinner("Retrieving..."):
        if mode == "Baseline":
            I, D = search_baseline(q, k)
        elif mode == "HyDE (1 hypo)":
            I, D = search_hyde(q, k, n=1)
        elif mode == "Multi-HyDE (2 hypos)":
            I, D = search_hyde(q, k, n=2)
        elif mode.startswith("Fusion"):
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=True)
            I = rrf_fuse_many([I_base, I_hyde], k=k)
        else:  # Hybrid
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=True)
            I_bm = bm25_search(q, k)
            I = rrf_fuse_many([I_base, I_hyde, I_bm], k=k)

    st.divider()
    st.markdown("### Retrieved corpus passages")
    for rank, pid in enumerate(I, start=1):
        title = passages[pid].get("title", f"P{pid}")
        text  = passages[pid].get("text", "")
        st.markdown(f"**#{rank} — [P{pid}] {title}**")
        st.markdown(highlight_snippet(text, q))
