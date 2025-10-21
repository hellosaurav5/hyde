from __future__ import annotations
from pathlib import Path
import json, re
import jsonlines

# HuggingFace datasets for HotpotQA (tiny slice)
from datasets import load_dataset

# ---- Paths
root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
runs_dir = root / "runs"
data_dir.mkdir(parents=True, exist_ok=True)
runs_dir.mkdir(parents=True, exist_ok=True)

# ---- Helpers (inline; no src imports)
def chunk_text(text: str, tok_len: int = 250, stride: int = 50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(1, tok_len - stride)
    while i < len(words):
        chunk = " ".join(words[i:i+tok_len])
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks

def build_passages_from_hotpot(in_jsonl: Path, out_jsonl: Path, tok_len: int = 250, stride: int = 50) -> None:
    """Robustly handle Hotpot context items as [title, [sents]] OR {'title':..., 'sentences':[...]}."""
    with open(in_jsonl, "r", encoding="utf-8") as f, jsonlines.open(out_jsonl, "w") as w:
        for line in f:
            ex = json.loads(line)
            ctx_list = ex.get("context", []) or []
            for ctx in ctx_list:
                title, sents = None, None
                if isinstance(ctx, dict):
                    title = ctx.get("title", "")
                    sents = ctx.get("sentences") or ctx.get("text") or []
                elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
                    title, sents = ctx[0], ctx[1]
                else:
                    continue

                # Normalize sentences/text
                if isinstance(sents, list):
                    text = " ".join(sents)
                else:
                    text = str(sents) if sents is not None else ""

                if not text.strip():
                    continue

                for ch in chunk_text(text, tok_len, stride):
                    w.write({"doc_id": title or "", "title": title or "", "text": ch})

def build_gold_from_answers(queries_jsonl: Path, passages_jsonl: Path, out_gold_jsonl: Path) -> None:
    # Heuristic: passage is relevant if it contains any gold answer string (case-insensitive)
    passages = list(jsonlines.open(passages_jsonl))
    # add pid by order
    for i, p in enumerate(passages):
        p["pid"] = i

    with open(queries_jsonl, "r", encoding="utf-8") as f, jsonlines.open(out_gold_jsonl, "w") as w:
        for line in f:
            ex = json.loads(line)
            qid = ex.get("_id", ex.get("id", ""))
            ans = ex.get("answer", "")
            answers = ans if isinstance(ans, list) else [ans]
            answers = [a for a in answers if isinstance(a, str) and a.strip()]
            rel_pids = []
            if answers:
                pat = re.compile("|".join(re.escape(a) for a in answers), flags=re.I)
                for p in passages:
                    if pat.search(p["text"]):
                        rel_pids.append(p["pid"])
            w.write({"qid": qid, "relevant_pids": sorted(set(rel_pids))})

# ---- Download a tiny HotpotQA slice
print("Downloading HotpotQA (distractor)...")
ds = load_dataset("hotpot_qa", "distractor")
