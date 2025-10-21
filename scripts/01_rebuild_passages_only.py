from __future__ import annotations
from pathlib import Path
import json, re
import jsonlines

root = Path(__file__).resolve().parents[1]
data_dir = root / "data"
runs_dir = root / "runs"
data_dir.mkdir(parents=True, exist_ok=True)
runs_dir.mkdir(parents=True, exist_ok=True)

train_path    = data_dir / "hotpot_train.jsonl"
valid_path    = data_dir / "hotpot_valid.jsonl"
passages_path = data_dir / "passages.jsonl"
gold_path     = runs_dir / "gold.jsonl"

def chunk_text(text: str, tok_len: int = 250, stride: int = 50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(1, tok_len - stride)
    while i < len(words):
        chunk = " ".join(words[i:i+tok_len])
        if chunk: chunks.append(chunk)
        i += step
    return chunks

def robust_iter_contexts(ex):
    ctx = ex.get("context", [])
    # dict-of-lists case: {"title":[...], "sentences":[[...], ...]}
    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles") or []
        sents_list = ctx.get("sentences") or []
        for i, title in enumerate(titles):
            sents = sents_list[i] if i < len(sents_list) else []
            text = " ".join(sents) if isinstance(sents, list) else (sents or "")
            if str(text).strip():
                yield title or "", str(text)
        return
    # list cases: [title, [sents]] OR {"title":..., "sentences":[...]}
    if isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, dict):
                title = item.get("title","")
                sents = item.get("sentences") or item.get("text") or []
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sents = item[0], item[1]
            else:
                continue
            text = " ".join(sents) if isinstance(sents, list) else (sents or "")
            if str(text).strip():
                yield title or "", str(text)

def write_passages(src_jsonl: Path, out_jsonl: Path, tok_len=250, stride=50, append=False):
    mode = "a" if append and out_jsonl.exists() else "w"
    n_examples = n_ctx = n_chunks = 0
    with open(src_jsonl, "r", encoding="utf-8") as f, jsonlines.open(out_jsonl, mode) as w:
        for line in f:
            ex = json.loads(line)
            n_examples += 1
            for title, text in robust_iter_contexts(ex):
                n_ctx += 1
                for ch in chunk_text(text, tok_len, stride):
                    n_chunks += 1
                    w.write({"doc_id": title, "title": title, "text": ch})
    print(f"[{src_jsonl.name}] examples={n_examples} contexts={n_ctx} chunks_written={n_chunks}")

def build_gold_from_answers(queries_jsonl: Path, passages_jsonl: Path, out_gold_jsonl: Path) -> None:
    passages = list(jsonlines.open(passages_jsonl))
    with open(queries_jsonl, "r", encoding="utf-8") as f, jsonlines.open(out_gold_jsonl, "w") as w:
        for line in f:
            ex = json.loads(line)
            qid = ex.get("_id") or ex.get("id") or ""
            ans = ex.get("answer","")
            answers = ans if isinstance(ans, list) else [ans]
            answers = [a for a in answers if isinstance(a, str) and a.strip()]
            rel_pids = []
            if answers:
                import re
                pat = re.compile("|".join(re.escape(a) for a in answers), flags=re.I)
                for i, p in enumerate(passages):
                    if pat.search(p["text"]):
                        rel_pids.append(i)
            w.write({"qid": qid, "relevant_pids": sorted(set(rel_pids))})

if __name__ == "__main__":
    import sys
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "full"
    if mode not in {"train", "full"}:
        raise ValueError("Mode must be 'train' or 'full' (default: full)")

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Missing train/valid jsonl. Run: python -m scripts.01_prepare_data")

    # (Re)start passages file
    if passages_path.exists():
        passages_path.unlink()

    if mode == "train":
        print("Building passages from TRAIN only …")
        write_passages(train_path, passages_path, tok_len=250, stride=50, append=False)
    else:  # full
        print("Building passages from TRAIN+VALID …")
        write_passages(train_path, passages_path, tok_len=250, stride=50, append=False)
        write_passages(valid_path, passages_path, tok_len=250, stride=50, append=True)

    print(f"Wrote passages to {passages_path}")

    # Gold is still built from VALID questions (evaluation set)
    build_gold_from_answers(valid_path, passages_path, gold_path)
    print(f"Wrote gold to {gold_path}")

