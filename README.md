# HYDE Demo (ACL’23) — Minimal, CPU-friendly

This repo is a compact, end-to-end implementation of **HyDE** (Hypothetical Document Embeddings) with:

* Dense baseline (E5)
* HyDE (single & multi-hypo)
* **Fusion (RRF)** and **Hybrid (BM25 + Dense + HyDE)**
* A Streamlit demo UI

Everything runs on **CPU** (Windows-friendly).

---

## Repo Layout

```
app/
  demo.py                  # Streamlit UI (Baseline / HyDE / Multi-HyDE / Fusion / Hybrid)
data/
  ...                      # Created by scripts (train/valid dumps, passages.jsonl)
runs/
  ...                      # Embeddings, index, meta, hyde cache, eval outputs
scripts/
  01_prepare_data.py       # One-time: download HotpotQA + dump JSONL splits
  01_rebuild_passages_only.py  # Build passages + gold (mode: train | full)
  02_build_index.py        # Embed passages (E5) and build FAISS or sklearn index
  03_eval_all.py           # Baseline / HyDE / Multi-HyDE / Fusion / Hybrid metrics
  04_report_plots.py       # (optional) CSV + bar charts for Recall@10 / MRR@10
requirements.txt
README.md
```

---

## 0) Environment Setup

Create a virtualenv and install requirements:

```bash
# from project root
python -m venv .venv
# PowerShell:   .\.venv\Scripts\Activate.ps1
# cmd/Git Bash: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Optional FAISS (CPU). If it fails, we automatically fall back to sklearn:
pip install -r requirements-extra-faiss.txt
```

**`requirements.txt`**

```txt
# Core DL stack (CPU)
torch==2.3.1
transformers==4.41.2
sentence-transformers==2.7.0

# Data + utilities
datasets==2.20.0
jsonlines==4.0.0
tqdm==4.66.4
pandas==2.2.2
matplotlib==3.8.4
joblib==1.3.2
scikit-learn==1.4.2

# Retrieval extras
rank-bm25==0.2.2

# App
streamlit==1.36.0

faiss-cpu==1.8.0.post1
```



---

## 1) Prepare the dataset (one-time)

Downloads a small HotpotQA “distractor” slice and writes JSONL files:

```bash
python -m scripts.01_prepare_data
```

Creates:

```
data/hotpot_train.jsonl
data/hotpot_valid.jsonl
```

---

## 2) Build passages & gold (choose a mode)

Use **train** for fair zero-shot eval; use **full** (train+valid) for best live demo.

### A) Fair eval (train-only index)

```bash
python -m scripts.01_rebuild_passages_only train
python -m scripts.02_build_index
python -m scripts.03_eval_all
```

### B) Demo (full index = train + valid)

```bash
python -m scripts.01_rebuild_passages_only full
python -m scripts.02_build_index
python -m scripts.03_eval_all   # optional: see numbers on the full index
```

Artifacts:

```
data/passages.jsonl
runs/passages.npy
runs/meta.json
runs/index.faiss  (if faiss-cpu installed)  OR  runs/index.sklearn
runs/gold.jsonl
runs/hyde_hypos.json   # HyDE generation cache
```

---

## 3) Run the demo app

```bash
streamlit run app/demo.py
```

**UI Modes**

* Baseline
* HyDE (1)
* Multi-HyDE (2)
* Fusion (Baseline + Multi-HyDE via RRF)
* Hybrid (BM25 + Baseline + Multi-HyDE)

The app shows the generated hypothetical passages (expandable) and the top-k retrieved passage titles.

---

## 4) Tips & Common Tasks

* **Switch modes?** After `01_rebuild_passages_only [train|full]`, re-run:

  ```bash
  python -m scripts.02_build_index
  ```
* **Fresh HyDE timing:** delete `runs/hyde_hypos.json` before `03_eval_all` to remove cache.
* **Latency:** `flan-t5-small`, `N_HYPOS_MULTI=2`, `HYPO_MAX_NEW=80` keep CPU runs snappy.
* **Fairness in report:** include results from **train-only** index (zero-shot) alongside **full** demo numbers.

---



## 5) Optional: Metrics CSV + Plots

Generate a quick table and bar charts (Recall@10 / MRR@10):

```bash
python -m scripts.04_report_plots
```

Outputs:

```
runs/report_metrics.csv
runs/recall10.png
runs/mrr10.png
```



