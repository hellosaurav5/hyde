from __future__ import annotations
import sys, json, time, hashlib, re
from pathlib import Path
import numpy as np
import jsonlines

# ========================
# Config (tweak if needed)
# ========================
ENCODER_MODEL = "intfloat/e5-base-v2"
HYDE_MODEL    = "google/flan-t5-small"   # faster on CPU
TOP_K         = 20
N_HYPOS_MULTI = 2                        # good speed/quality tradeoff
HYPO_MAX_NEW  = 80    #make it 60 to fasten                   # shorter, crisper hypos
HYPO_TEMP     = 0.0
HYPO_TOP_P    = 0.9
DEVICE        = "cpu"                    # keep CPU for Windows unless you have CUDA

root       = Path(__file__).resolve().parents[1]
val_path   = root / "data" / "hotpot_valid.jsonl"
faiss_path = root / "runs" / "index.faiss"
sk_path    = root / "runs" / "index.sklearn"
meta_path  = root / "runs" / "meta.json"
gold_path  = root / "runs" / "gold.jsonl"
runs_dir   = root / "runs"
runs_dir.mkdir(exist_ok=True, parents=True)

# HyDE cache on disk (so repeated runs are fast)
HYPO_CACHE  = runs_dir / "hyde_hypos.json"

# ============================
# Inline index wrapper/loader
# ============================
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

def load_index_inline(path_faiss: str, path_sklearn: str) -> _Index:
    p_faiss = Path(path_faiss)
    if p_faiss.exists():
        import faiss  # type: ignore
        return _Index(faiss_index=faiss.read_index(str(p_faiss)))
    else:
        import joblib
        return _Index(sklearn_nn=joblib.load(str(path_sklearn)))

# ========================
# Minimal retrieval metrics
# ========================
def recall_at_k(gold: dict[str, list[int]], retrieved: dict[str, list[int]], k: int=10) -> float:
    hits = []
    for qid, rel in gold.items():
        ret = retrieved.get(qid, [])[:k]
        hits.append(int(any(pid in set(ret) for pid in rel)))
    return float(np.mean(hits)) if hits else 0.0

def mrr_at_k(gold: dict[str, list[int]], retrieved: dict[str, list[int]], k: int=10) -> float:
    vals = []
    for qid, rel in gold.items():
        ret = retrieved.get(qid, [])[:k]
        rr = 0.0
        for rank, pid in enumerate(ret, start=1):
            if pid in set(rel): rr = 1.0/rank; break
        vals.append(rr)
    return float(np.mean(vals)) if vals else 0.0

# ===========
# RRF fusion
# ===========
def rrf_fuse(ids_a, ids_b, k=TOP_K, K=60):
    scores = {}
    for r, pid in enumerate(ids_a, 1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (K + r)
    for r, pid in enumerate(ids_b, 1):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (K + r)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]

def rrf_fuse_many(lists, k=TOP_K, K=60):
    scores = {}
    for lst in lists:
        for r, pid in enumerate(lst, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0/(K + r)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:k]

# ============================
# Encoder & HyDE generator
# ============================
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

_encoder = SentenceTransformer(ENCODER_MODEL, device=DEVICE)

hf_tokenizer = AutoTokenizer.from_pretrained(HYDE_MODEL)
hf_model     = AutoModelForSeq2SeqLM.from_pretrained(HYDE_MODEL).to(DEVICE)

PROMPT = (
  "Write a concise, factual passage (3–5 sentences) that would directly answer the question. "
  "Prefer key entities, dates, and definitions; avoid speculation.\n\n"
  "Question: {q}\nHypothetical passage:"
)

@torch.inference_mode()
def gen_hypo(question: str, max_new_tokens: int=HYPO_MAX_NEW, temperature: float=HYPO_TEMP, top_p: float=HYPO_TOP_P) -> str:
    x = hf_tokenizer(PROMPT.format(q=question), return_tensors="pt").to(DEVICE)
    y = hf_model.generate(**x, max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=temperature, top_p=top_p)
    return hf_tokenizer.decode(y[0], skip_special_tokens=True).strip()

def enc_query(text: str) -> np.ndarray:
    return _encoder.encode(["query: " + text], normalize_embeddings=True)

def enc_query_vec(text: str) -> np.ndarray:
    return _encoder.encode("query: " + text, normalize_embeddings=True)

# ---------------
# HyDE cache I/O
# ---------------
if HYPO_CACHE.exists():
    _HYPO_CACHE = json.loads(HYPO_CACHE.read_text(encoding="utf-8"))
else:
    _HYPO_CACHE = {}

def _cache_key(qid: str|None, qtext: str, n: int) -> str:
    if qid:
        return f"{qid}::n={n}"
    h = hashlib.md5(qtext.strip().encode("utf-8")).hexdigest()[:10]
    return f"hash:{h}::n={n}"

def _cache_get(key: str):
    return _HYPO_CACHE.get(key)

def _cache_put(key: str, hypos: list[str]):
    _HYPO_CACHE[key] = hypos

def _cache_flush():
    HYPO_CACHE.write_text(json.dumps(_HYPO_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")

# ===================
# Retrieval functions
# ===================
def baseline_search(question: str, idx: _Index, k: int=TOP_K):
    t0 = time.time()
    qv = enc_query(question)
    I, D = idx.search(qv.astype("float32"), k)
    return {"mode": "baseline", "pids": I[0].tolist(), "scores": D[0].tolist(),
            "latency_ms": int((time.time()-t0)*1000), "hypos": []}

def hyde_search(question: str, idx: _Index, k: int=TOP_K, n_hypos: int=1, qid: str|None=None):
    t0 = time.time()
    key = _cache_key(qid, question, n_hypos)
    hypos = _cache_get(key)
    if not hypos:
        hypos = [gen_hypo(question) for _ in range(n_hypos)]
        _cache_put(key, hypos)
    hv = _encoder.encode(["query: " + h for h in hypos], normalize_embeddings=True).mean(axis=0)
    I, D = idx.search(hv[None, :].astype("float32"), k)
    return {"mode": f"hyde_n{n_hypos}", "pids": I[0].tolist(), "scores": D[0].tolist(),
            "latency_ms": int((time.time()-t0)*1000), "hypos": hypos}

# =============================
# BM25 lexical retriever (quick)
# =============================
from rank_bm25 import BM25Okapi

# Load passages text for BM25
pass_texts = []
with jsonlines.open(root / "data" / "passages.jsonl") as r:
    for rec in r:
        pass_texts.append(rec["text"])

def bm25_tokenize(s: str):
    # simple word tokenizer, lowercase, alnum
    return re.findall(r"\w+", s.lower())

bm25 = BM25Okapi([bm25_tokenize(t) for t in pass_texts])

def bm25_search(question: str, k: int = TOP_K*2):
    scores = bm25.get_scores(bm25_tokenize(question))
    idx = np.argsort(-np.array(scores))[:k]
    return idx.tolist()

# ==========
# Load data
# ==========
if not val_path.exists():
    raise FileNotFoundError(f"Missing {val_path}. Run: python -m scripts.01_prepare_data")
if not (faiss_path.exists() or sk_path.exists()):
    raise FileNotFoundError("Missing index. Run: python -m scripts.02_build_index")

# Load queries (robust to id/_id)
queries = []
with open(val_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        ex = json.loads(line)
        qid = ex.get("_id") or ex.get("id") or f"q{i}"
        question = ex["question"]
        answers  = ex.get("answer", "")
        queries.append({"qid": qid, "question": question, "answers": answers})

# Load gold (heuristic string-match)
gold = {}
with jsonlines.open(gold_path) as r:
    for rec in r:
        gold[str(rec["qid"])] = rec.get("relevant_pids", [])

idx = load_index_inline(str(faiss_path), str(sk_path))

# ==========================
# Runner and pretty report
# ==========================
def run_mode(mode: str, n_hypos: int=1, k: int=TOP_K, out_jsonl: Path|None=None):
    rows = []
    for ex in queries:
        qid, q = ex["qid"], ex["question"]
        if mode == "baseline":
            res = baseline_search(q, idx, k=k)
        elif mode == "hyde":
            res = hyde_search(q, idx, k=k, n_hypos=1, qid=qid)
        elif mode == "multi":
            res = hyde_search(q, idx, k=k, n_hypos=n_hypos, qid=qid)
        elif mode == "fuse":
            base = baseline_search(q, idx, k=k)
            hyde = hyde_search(q, idx, k=k, n_hypos=n_hypos, qid=qid)
            fused_pids = rrf_fuse(base["pids"], hyde["pids"], k=k)
            res = {"mode": f"fuse_n{n_hypos}", "pids": fused_pids, "scores": [],
                   "latency_ms": max(base["latency_ms"], hyde["latency_ms"]),
                   "hypos": hyde["hypos"]}
        elif mode == "hybrid":
            base = baseline_search(q, idx, k=k)
            hyde = hyde_search(q, idx, k=k, n_hypos=n_hypos, qid=qid)
            bm   = bm25_search(q, k=k)  # lexical candidates
            fused_pids = rrf_fuse_many([base["pids"], hyde["pids"], bm], k=k)
            res = {"mode": f"hybrid_n{n_hypos}", "pids": fused_pids, "scores": [],
                   "latency_ms": max(base["latency_ms"], hyde["latency_ms"]),
                   "hypos": hyde["hypos"]}
        else:
            raise ValueError(mode)
        rows.append({"qid": qid, **res})

    if out_jsonl is not None:
        with jsonlines.open(out_jsonl, "w") as w:
            for r in rows: w.write(r)

    ret = {
        "retrieved": {r["qid"]: r["pids"] for r in rows},
        "latency_ms_avg": float(np.mean([r["latency_ms"] for r in rows]))
    }
    _cache_flush()  # persist hypos for next runs
    return ret

def report(name, ret):
    R = recall_at_k(gold, ret["retrieved"], k=TOP_K)
    M = mrr_at_k(gold, ret["retrieved"], k=TOP_K)
    print(f"{name:>18}  Recall@{TOP_K}: {R:.3f}   MRR@{TOP_K}: {M:.3f}   Avg Latency(ms): {ret['latency_ms_avg']:.0f}")

# =====
# Main
# =====
if __name__ == "__main__":
    try:
        print("Running Baseline ...")
        ret_base  = run_mode("baseline", out_jsonl=runs_dir / "retrieval_baseline.jsonl")
        print("Running HyDE (1 hypo) ...")
        ret_hyde  = run_mode("hyde", out_jsonl=runs_dir / "retrieval_hyde.jsonl")
        print(f"Running Multi-HyDE (n={N_HYPOS_MULTI}) ...")
        ret_multi = run_mode("multi", n_hypos=N_HYPOS_MULTI, out_jsonl=runs_dir / "retrieval_multi.jsonl")
        print(f"Running Fusion (Baseline + Multi-HyDE n={N_HYPOS_MULTI}) ...")
        ret_fuse  = run_mode("fuse", n_hypos=N_HYPOS_MULTI, out_jsonl=runs_dir / "retrieval_fuse.jsonl")
        print(f"Running Hybrid (BM25 + Baseline + Multi-HyDE n={N_HYPOS_MULTI}) ...")
        ret_hybrid = run_mode("hybrid", n_hypos=N_HYPOS_MULTI, out_jsonl=runs_dir / "retrieval_hybrid.jsonl")

        print("\nResults")
        report("Baseline", ret_base)
        report("HyDE (1)", ret_hyde)
        report(f"Multi-HyDE ({N_HYPOS_MULTI})", ret_multi)
        report("Fusion", ret_fuse)
        report("Hybrid", ret_hybrid)
        print("\nJSONL outputs in runs/.")
    finally:
        _cache_flush()
