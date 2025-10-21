from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import jsonlines
from tqdm import tqdm

# HF transformers for E5 encoder
import torch
from transformers import AutoTokenizer, AutoModel

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
runs_dir = root / "runs"
runs_dir.mkdir(exist_ok=True, parents=True)

passages_path = data_dir / "passages.jsonl"
emb_path  = runs_dir / "passages.npy"
meta_path = runs_dir / "meta.json"
faiss_path = runs_dir / "index.faiss"
sk_path    = runs_dir / "index.sklearn"

ENCODER_MODEL = "intfloat/e5-base-v2"
BATCH_SIZE = 64  # lower to 16/32 if RAM is tight

# --------- Load passages ----------
passages = list(jsonlines.open(str(passages_path)))
texts   = [p["text"] for p in passages]
titles  = [p.get("title","") for p in passages]
print(f"Loaded {len(texts)} passages")

# --------- Encoder (HF-only; CPU; avoid meta tensors) ----------
tok = AutoTokenizer.from_pretrained(ENCODER_MODEL, use_fast=True)
mdl = AutoModel.from_pretrained(ENCODER_MODEL, low_cpu_mem_usage=False, device_map=None)
mdl.eval()

@torch.inference_mode()
def encode_batch(batch_texts):
    # E5 expects "passage: ..." for docs
    batch = tok(["passage: " + t for t in batch_texts],
                padding=True, truncation=True, max_length=512, return_tensors="pt")
    out = mdl(**batch)
    emb = out.last_hidden_state.mean(dim=1)          # [B, H]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy().astype("float32")

# --------- Embed all passages ----------
all_embs = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encode_passages"):
    all_embs.append(encode_batch(texts[i:i+BATCH_SIZE]))
embs = np.vstack(all_embs)
np.save(str(emb_path), embs)
json.dump([{"pid": i, "title": titles[i]} for i in range(len(texts))],
          open(meta_path, "w", encoding="utf-8"))
print(f"Saved embeddings -> {emb_path}  shape={embs.shape}")
print(f"Saved meta -> {meta_path}")

# --------- Build index (FAISS if available; else sklearn) ----------
class IndexWrapper:
    def __init__(self, faiss_index=None, sklearn_nn=None):
        self.faiss_index = faiss_index
        self.sklearn_nn = sklearn_nn
    def search(self, qvecs: np.ndarray, k: int = 10):
        if self.faiss_index is not None:
            D, I = self.faiss_index.search(qvecs.astype("float32"), k)
            return I, D
        D, I = self.sklearn_nn.kneighbors(qvecs, n_neighbors=k, return_distance=True)
        return I, D

def build_index(embs: np.ndarray) -> IndexWrapper:
    try:
        import faiss  # type: ignore
        d = embs.shape[1]
        index = faiss.IndexHNSWFlat(d, 64)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 128
        index.add(embs.astype("float32"))
        return IndexWrapper(faiss_index=index)
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(embs)
        return IndexWrapper(sklearn_nn=nn)

def save_index(idx: IndexWrapper, path_faiss: str, path_sklearn: str):
    if idx.faiss_index is not None:
        import faiss  # type: ignore
        faiss.write_index(idx.faiss_index, path_faiss)
        print(f"Saved FAISS index -> {path_faiss}")
    else:
        import joblib
        joblib.dump(idx.sklearn_nn, path_sklearn)
        print(f"Saved sklearn index -> {path_sklearn}")

idx = build_index(embs)
save_index(idx, str(faiss_path), str(sk_path))
print("Done.")
