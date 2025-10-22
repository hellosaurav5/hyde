# app/demo.py — HyDE demo with Gemini 2.0 Flash (google.genai Client) + local fallback, Fusion/Hybrid
from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import jsonlines
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# ========== UI ==========
st.set_page_config(page_title="HyDE (ACL'23) — Demo", layout="wide")
st.title("HyDE (ACL'23) — Minimal Demo")

mode = st.radio(
    "Mode",
    ["Baseline", "HyDE (1 hypo)", "Multi-HyDE (2 hypos)",
     "Fusion (Baseline+Multi-HyDE)", "Hybrid (BM25+Baseline+Multi-HyDE)"],
    horizontal=True,
)
k = st.slider("Top-k", 5, 30, 10)
q = st.text_input("Question", "Who wrote the Federalist Papers and what was the purpose?")
show_hypos_ui = st.checkbox("Show hypothetical passages", value=True)

# ========== Paths / corpus ==========
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

# ========== Index loader ==========
class _Index:
    def __init__(self, faiss_index=None, sklearn_nn=None):
        self._faiss = faiss_index; self._sk = sklearn_nn
    def search(self, qvecs: np.ndarray, k: int = 10):
        if self._faiss is not None:
            D, I = self._faiss.search(qvecs.astype("float32"), k); return I, D
        D, I = self._sk.kneighbors(qvecs, n_neighbors=k, return_distance=True); return I, D

def load_index_inline(path_faiss: Path, path_sklearn: Path) -> _Index:
    if path_faiss.exists():
        import faiss
        return _Index(faiss_index=faiss.read_index(str(path_faiss)))
    import joblib
    if not path_sklearn.exists():
        st.error("No index found. Run: python -m scripts.02_build_index")
        st.stop()
    return _Index(sklearn_nn=joblib.load(str(path_sklearn)))

index = load_index_inline(root / "runs" / "index.faiss", root / "runs" / "index.sklearn")

# ========== E5 encoder (HF, CPU) ==========
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

# ========== Local HyDE fallback ==========
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

# ========== Gemini 2.0 Flash (google.genai Client) ==========
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

client = None
GEN_CFG = None
try:
    from google import genai
    from google.genai import types
    if API_KEY:
        client = genai.Client(api_key=API_KEY)

        SYSTEM_INSTRUCTION = (
            "You write short, factual, Wikipedia-style passages that answer questions concisely. "
            "Avoid any sexual content, profanity, violence, or unsafe instructions. Neutral tone."
        )

        GEN_CFG = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0, top_p=0.9, max_output_tokens=120,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                                    threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ],
        )
except Exception:
    client = None
    GEN_CFG = None

# Toggle + live status
USE_GEMINI = st.checkbox("Use Gemini 2.0 Flash for HyDE generation", value=bool(client), key="use_gemini")
api_ok = bool(client)
engine = "Gemini 2.0 Flash (google.genai)" if (USE_GEMINI and api_ok) else "Local flan-T5-small"
st.caption(f"HyDE generator: **{engine}**  |  API key: {'OK' if api_ok else 'MISSING'}  |  model: {'gemini-2.0-flash' if api_ok else '(none)'}")

# One-click test to verify Gemini is active
if api_ok and st.button("Test Gemini now"):
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Say OK in one word."],
            config=GEN_CFG,
        )
        st.success(f"Gemini response: {(getattr(resp,'text','') or '').strip() or '(no text)'}")
    except Exception as e:
        st.error(f"Gemini test failed: {e}")

def _extract_gemini_text(resp) -> str:
    txt = (getattr(resp, "text", None) or "").strip()
    if txt:
        return txt
    # fallback: assemble text from candidates.parts
    pieces = []
    for c in getattr(resp, "candidates", []) or []:
        content = getattr(c, "content", None)
        if content and getattr(content, "parts", None):
            for p in content.parts:
                pt = getattr(p, "text", None)
                if pt:
                    pieces.append(pt)
    return " ".join(pieces).strip()

def gen_hypo_with_gemini(question: str) -> str:
    if not (USE_GEMINI and client and GEN_CFG):
        return ""
    prompt = PROMPT.format(q=question)
    # retry up to 2 times
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=GEN_CFG,
            )
            txt = _extract_gemini_text(resp)
            if txt:
                return txt
            if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                st.warning(f"Gemini blocked/empty: {resp.prompt_feedback}")
        except Exception as e:
            if attempt == 1:
                st.warning(f"Gemini error: {e}; using local fallback.")
    return ""

# ========== Engine-aware cache (clear on toggle) ==========
if "hyde_cache" not in st.session_state:
    st.session_state["hyde_cache"] = {}
if "prev_use_gemini" not in st.session_state:
    st.session_state["prev_use_gemini"] = USE_GEMINI
elif st.session_state["prev_use_gemini"] != USE_GEMINI:
    st.session_state["hyde_cache"].clear()
    st.session_state["prev_use_gemini"] = USE_GEMINI

def cache_key(question: str, n: int) -> str:
    h = hashlib.md5(question.strip().encode("utf-8")).hexdigest()[:10]
    engine_tag = "gem20" if (USE_GEMINI and api_ok) else "loc"
    return f"{h}::n={n}::engine={engine_tag}"

def gen_hypo(question: str) -> str:
    if USE_GEMINI and api_ok:
        txt = gen_hypo_with_gemini(question)
        if txt:
            return txt
        else:
            st.info("Falling back to local flan-T5 for this query.")
    return _local_hypo(PROMPT.format(q=question))

def get_hypos(question: str, n: int):
    key = cache_key(question, n)
    cache = st.session_state["hyde_cache"]
    if key not in cache:
        hypos = []
        for _ in range(n):
            h = gen_hypo(question)
            if h and h.strip():
                hypos.append(h.strip())
        if not hypos:
            hypos = [_local_hypo(PROMPT.format(q=question))]
        cache[key] = hypos
    return cache[key]

# ========== BM25 / Fusion ==========
pass_texts = [rec["text"] for rec in passages]
def tok_simple(s: str): return re.findall(r"\w+", s.lower())
bm25 = BM25Okapi([tok_simple(t) for t in pass_texts])

def bm25_search(question: str, k: int = 20):
    scores = bm25.get_scores(tok_simple(question)); idx = np.argsort(-np.array(scores))[:k]
    return idx.tolist()

def rrf_fuse_many(lists, k=10, K=60):
    scores = {}
    for lst in lists:
        for r, pid in enumerate(lst, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0/(K+r)
    return [pid for pid,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)][:k]

# ========== Search funcs ==========
def search_baseline(question: str, k: int):
    qv = e5_encode(["query: " + question]); I, D = index.search(qv.astype("float32"), k)
    return I[0], D[0]

def search_hyde(question: str, k: int, n: int = 1, show_hypos: bool = True):
    hypos = get_hypos(question, n)
    if show_hypos:
        with st.expander("Hypothetical passages", expanded=False):
            for i, h in enumerate(hypos, 1): st.code(f"[{i}] {h}")
    hv = e5_encode(["query: " + h for h in hypos]).mean(axis=0)
    I, D = index.search(hv[None,:].astype("float32"), k)
    return I[0], D[0]

# ========== Action ==========
if st.button("Search"):
    with st.spinner("Retrieving..."):
        if mode == "Baseline":
            I, D = search_baseline(q, k)
        elif mode == "HyDE (1 hypo)":
            I, D = search_hyde(q, k, n=1, show_hypos=show_hypos_ui)
        elif mode == "Multi-HyDE (2 hypos)":
            I, D = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
        elif mode.startswith("Fusion"):
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
            I = rrf_fuse_many([I_base, I_hyde], k=k)
        else:
            I_base, _ = search_baseline(q, k)
            I_hyde, _ = search_hyde(q, k, n=2, show_hypos=show_hypos_ui)
            I_bm = bm25_search(q, k)
            I = rrf_fuse_many([I_base, I_hyde, I_bm], k=k)

    st.divider()
    st.markdown("### Retrieved corpus passages")
    for rank, pid in enumerate(I, start=1):
        title = passages[pid].get("title", f"P{pid}")
        text  = passages[pid].get("text", "")
        st.markdown(f"**#{rank} — [P{pid}] {title}**")
        st.markdown(highlight_snippet(text, q))
