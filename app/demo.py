# app/demo.py — HyDE demo (simple) with Gemini 2.0 Flash via google.genai + local fallback
from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import jsonlines
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="HyDE (ACL'23) — Demo", layout="wide")
st.title("HyDE (ACL'23) — Minimal Demo")

mode = st.radio(
    "Mode",
    ["Baseline", "HyDE (1 hypo)", "Multi-HyDE (2 hypos)",
     "Fusion (Baseline+Multi-HyDE)", "Hybrid (BM25+Baseline+Multi-HyDE)"],
    horizontal=True,
)
k = st.slider("Top-k", 5, 30, 10)
q = st.text_input("Question", "Iqaluit Airport and Canadian North are based out of what country?")
show_hypos_ui = st.checkbox("Show hypothetical passages", value=True)

# Sidebar: paste your Google API key (no env var needed)
st.sidebar.header("Gemini Settings")
if "GKEY" not in st.session_state:
    st.session_state.GKEY = os.environ.get("GOOGLE_API_KEY", "")
gkey_input = st.sidebar.text_input("GOOGLE_API_KEY", value=st.session_state.GKEY, type="password")
if st.sidebar.button("Use This Key"):
    st.session_state.GKEY = gkey_input.strip()

# =========================
# Paths / corpus
# =========================
root = Path(__file__).resolve().parents[1]
meta_path = root / "runs" / "meta.json"
passages_path = root / "data" / "passages.jsonl"

if not meta_path.exists():
    st.error(f"Missing {meta_path}. Build index first: python -m scripts.02_build_index")
    st.stop()
if not passages_path.exists():
    st.error(f"Missing {passages_path}. Run: python -m scripts.01_prepare_data")
    st.stop()

meta = json.load(open(meta_path, "r", encoding="utf-8"))
passages = list(jsonlines.open(str(passages_path)))

def highlight_snippet(text: str, query: str, max_chars: int = 400) -> str:
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    s = text
    for t in terms:
        s = re.sub(fr"(?i)\b{re.escape(t)}\b", r"**\g<0>**", s)
    return (s[:max_chars] + "…") if len(s) > max_chars else s

# =========================
# Index loader (faiss or sklearn)
# =========================
# =========================
# Index loader (faiss or sklearn) + metric detection
# =========================
class _Index:
    def __init__(self, faiss_index=None, sklearn_nn=None):
        self._faiss = faiss_index
        self._sk = sklearn_nn
        if faiss_index is not None:
            import faiss
            self.backend = "faiss"
            mt = getattr(faiss_index, "metric_type", faiss.METRIC_INNER_PRODUCT)
            self.metric = "ip" if mt == faiss.METRIC_INNER_PRODUCT else "l2"
        else:
            self.backend = "sklearn-cosine"
            self.metric = "cosine"

    def search(self, qvecs: np.ndarray, k: int = 10):
        if self._faiss is not None:
            D, I = self._faiss.search(qvecs.astype("float32"), k)  # FAISS: (distances/sims, indices)
            return I, D
        # sklearn NearestNeighbors (cosine): returns distances (smaller = better)
        D, I = self._sk.kneighbors(qvecs, n_neighbors=k, return_distance=True)
        return I, D

def load_index_inline(path_faiss: Path, path_sklearn: Path) -> _Index:
    if path_faiss.exists():
        import faiss
        return _Index(faiss_index=faiss.read_index(str(path_faiss)))
    else:
        import joblib
        if not path_sklearn.exists():
            st.error("No index found. Run: python -m scripts.02_build_index")
            st.stop()
        return _Index(sklearn_nn=joblib.load(str(path_sklearn)))

    
index = load_index_inline(root / "runs" / "index.faiss", root / "runs" / "index.sklearn")

# --- normalize scores to a positive, "higher is better" similarity ---
def _sim_scores(D_row):
    """
    Returns a cosine-like similarity in [-1, 1] (often ~[0.5, 0.9] for good hits):
      - FAISS (inner product): already similarity -> return as-is.
      - FAISS (L2): FAISS returns *squared* L2 distances; for unit vectors,
                    cos_sim = 1 - (L2^2)/2  -> use 1 - 0.5 * D.
      - sklearn (cosine distance): distance = 1 - cos_sim -> use 1 - distance.
    """
    import numpy as np
    D = np.asarray(D_row, dtype=np.float32)

    if index.backend == "faiss":
        # If the index metric is available, it will be set to "ip" or "l2" in our loader.
        if getattr(index, "metric", "ip") == "ip":
            S = D  # already inner-product similarity
        else:  # "l2"
            # Convert squared L2 distance to cosine-like similarity
            S = 1.0 - 0.5 * D
    else:
        # sklearn NearestNeighbors(metric='cosine') -> distances; convert to similarity
        S = 1.0 - D

    # Clamp to [-1, 1] just for neat display
    S = np.clip(S, -1.0, 1.0)
    return S


# =========================
# Encoder (E5 via HF, CPU)
# =========================
@st.cache_resource(show_spinner=False)
def get_e5():
    tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", use_fast=True)
    mdl = AutoModel.from_pretrained("intfloat/e5-base-v2", low_cpu_mem_usage=False, device_map=None)
    mdl.eval(); return tok, mdl
tok_e5, mdl_e5 = get_e5()

@torch.inference_mode()
def e5_encode(texts: list[str]) -> np.ndarray:
    batch = tok_e5(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = mdl_e5(**batch)
    emb = out.last_hidden_state.mean(dim=1)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

# =========================
# Local HyDE fallback (flan-t5-small)
# =========================
@st.cache_resource(show_spinner=False)
def get_local_hyde():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    mdl.eval(); return tok, mdl
tok_hyde, mdl_hyde = get_local_hyde()

def _local_hypo(prompt: str) -> str:
    x = tok_hyde(prompt, return_tensors="pt")
    y = mdl_hyde.generate(**x, max_new_tokens=80, do_sample=False)
    return tok_hyde.decode(y[0], skip_special_tokens=True).strip()

PROMPT = (
    "Write a short, factual paragraph (2–4 sentences) in Wikipedia style that could answer the question. "
    "Use real entity names and concrete facts. Include the specific city if relevant. "
    "Do NOT restate the question. Do NOT invent facts. Keep it concise.\n\n"
    "Question: {q}\nPassage:"
)

# =========================
# Gemini 2.0 Flash via google.genai (simple)
# =========================
client = None
GEN_CFG = None
try:
    from google import genai
    from google.genai import types
    if st.session_state.GKEY:
        client = genai.Client(api_key=st.session_state.GKEY)
        GEN_CFG = types.GenerateContentConfig(
            system_instruction=(
                "You write short, factual, Wikipedia-style passages that answer questions concisely. "
                "Avoid sexual content, profanity, violence, or unsafe instructions. Neutral tone."
            ),
            temperature=0.0, top_p=0.9, max_output_tokens=120,
            # safety settings kept minimal & default-friendly; adjust if needed
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ],
        )
except Exception as e:
    client = None
    GEN_CFG = None
    st.sidebar.warning(f"google.genai not available: {e}")

# Toggle + status
USE_GEMINI = st.checkbox("Use Gemini 2.0 Flash for HyDE generation", value=bool(client), key="use_gemini")
api_ok = bool(client)
engine = "Gemini 2.0 Flash" if (USE_GEMINI and api_ok) else "Local flan-T5-small"
st.caption(f"HyDE generator: **{engine}**  |  API key: {'OK' if api_ok else 'MISSING'}  |  model: {'gemini-2.0-flash' if api_ok else '(none)'}")

# Quick test button (verifies key + connectivity)
if api_ok and st.sidebar.button("Test Gemini now"):
    try:
        r = client.models.generate_content(model="gemini-2.0-flash", contents=["Say OK in one word."])
        st.sidebar.success(f"Gemini: {(getattr(r,'text','') or '').strip() or '(no text)'}")
    except Exception as e:
        st.sidebar.error(f"Gemini test failed: {e}")

def _extract_gemini_text(resp) -> str:
    txt = (getattr(resp, "text", None) or "").strip()
    if txt:
        return txt
    pieces = []
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                if getattr(p, "text", None):
                    pieces.append(p.text)
    return " ".join(pieces).strip()

def gen_hypo_with_gemini(question: str) -> str:
    if not (USE_GEMINI and client):
        return ""
    prompt = PROMPT.format(q=question)
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=GEN_CFG,
        )
        return _extract_gemini_text(resp)
    except Exception as e:
        st.info(f"Gemini error: {e}. Falling back to local.")
        return ""

# =========================
# Engine-aware cache (clear on toggle)
# =========================
if "hyde_cache" not in st.session_state:
    st.session_state["hyde_cache"] = {}
if "prev_use_gemini" not in st.session_state:
    st.session_state["prev_use_gemini"] = USE_GEMINI
elif st.session_state["prev_use_gemini"] != USE_GEMINI:
    st.session_state["hyde_cache"].clear()
    st.session_state["prev_use_gemini"] = USE_GEMINI

def cache_key(question: str, n: int) -> str:
    h = hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:10]
    tag = "gem20" if (USE_GEMINI and api_ok) else "loc"
    return f"{h}::n={n}::engine={tag}"

def gen_hypo(question: str) -> str:
    if USE_GEMINI and api_ok:
        t = gen_hypo_with_gemini(question)
        if t:
            return t
    return _local_hypo(PROMPT.format(q=question))

def get_hypos(question: str, n: int):
    key = cache_key(question, n)
    cache = st.session_state["hyde_cache"]
    if key not in cache:
        hypos = []
        for _ in range(n):
            h = gen_hypo(question)
            if h.strip():
                hypos.append(h.strip())
        if not hypos:
            hypos = [_local_hypo(PROMPT.format(q=question))]
        cache[key] = hypos
    return cache[key]

# =========================
# BM25 / Fusion helpers
# =========================
pass_texts = [rec["text"] for rec in passages]
def tok_simple(s: str): return re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok_simple(t) for t in pass_texts])

def bm25_search(question: str, k: int = 20):
    scores_all = bm25.get_scores(tok_simple(question))  # raw BM25
    idx = np.argsort(-np.array(scores_all))[:k]
    return idx.tolist(), np.array(scores_all)[idx]

def rrf_fuse_many(lists, k=10, K=60):
    """
    lists: list of rank lists (each is a list of pids)
    returns: (ranked_ids, score_map) where score_map[pid] is the fused score
    """
    scores = {}
    for lst in lists:
        for r, pid in enumerate(lst, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (K + r)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ids = [pid for pid, _ in ranked]
    score_map = {pid: sc for pid, sc in ranked}
    return ids, score_map

# =========================
# Search functions
# =========================
def search_baseline(question: str, k: int):
    qv = e5_encode(["query: " + question])
    I, D = index.search(qv.astype("float32"), k)
    S = _sim_scores(D[0])
    return I[0], S

def search_hyde(question: str, k: int, n: int = 1, show_hypos: bool = True):
    hypos = get_hypos(question, n)
    if show_hypos:
        with st.expander("Hypothetical passages", expanded=False):
            for i, h in enumerate(hypos, 1):
                st.code(f"[{i}] {h}")
    hv = e5_encode(["query: " + h for h in hypos]).mean(axis=0)
    I, D = index.search(hv[None, :].astype("float32"), k)
    S = _sim_scores(D[0])
    return I[0], S


def _normalize_score_map(score_map: dict[int, float]) -> dict[int, float]:
    import numpy as np
    if not score_map: return {}
    vals = np.array(list(score_map.values()), dtype=np.float32)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-9:
        return {pid: 1.0 for pid in score_map}  # all equal
    return {pid: float((s - vmin) / (vmax - vmin)) for pid, s in score_map.items()}


# =========================
# Action
# =========================
if st.button("Search"):
    with st.spinner("Retrieving..."):
        if mode == "Baseline":
            I, S = search_baseline(q, k)
            score_for = {pid: float(S[i]) for i, pid in enumerate(I)}
            score_label = "cosine-like sim"
        elif mode == "HyDE (1 hypo)":
            I, S = search_hyde(q, k, n=1, show_hypos=show_hypos_ui)
            score_for = {pid: float(S[i]) for i, pid in enumerate(I)}
            score_label = "cosine-like sim"
        elif mode == "Multi-HyDE (2 hypos)":
            I, S = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
            score_for = {pid: float(S[i]) for i, pid in enumerate(I)}
            score_label = "cosine-like sim"
        elif mode.startswith("Fusion"):
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
            I, rrf_scores = rrf_fuse_many([I_base, I_hyde], k=k)
            score_for = _normalize_score_map(rrf_scores)
            score_label = "RRF (normalized 0–1)"
        else:  # Hybrid
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
            I_bm, _ = bm25_search(q, k)
            I, rrf_scores = rrf_fuse_many([I_base, I_hyde, I_bm], k=k)
            score_for = _normalize_score_map(rrf_scores)
            score_label = "RRF (normalized 0–1)"

    st.divider()
    st.markdown("### Retrieved corpus passages")
    for rank, pid in enumerate(I, start=1):
        title = passages[pid].get("title", f"P{pid}")
        text  = passages[pid].get("text", "")
        score = score_for.get(pid, float("nan"))
        st.markdown(f"**#{rank} — [P{pid}] {title}**  \n*{score_label}:* `{score_for.get(pid, float('nan')):.4f}`")
        st.markdown(highlight_snippet(text, q))

