from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
import google.generativeai as genai

# --------------------
# CLI
# --------------------
p = argparse.ArgumentParser(description="Prewarm HyDE hypos with Gemini 2.5 Flash")
p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model id")
p.add_argument("--n", type=int, default=2, help="hypos per question")
p.add_argument("--limit", type=int, default=0, help="only process first N queries (0 = all)")
p.add_argument("--max-tokens", type=int, default=120)
p.add_argument("--temp", type=float, default=0.0)
p.add_argument("--top-p", type=float, default=0.9)
p.add_argument("--timeout", type=float, default=45.0)
p.add_argument("--retries", type=int, default=3)
args = p.parse_args()

# --------------------
# Paths / config
# --------------------
root       = Path(__file__).resolve().parents[1]
val_path   = root / "data" / "hotpot_valid.jsonl"
cache_path = root / "runs" / "hyde_hypos.json"
cache_path.parent.mkdir(parents=True, exist_ok=True)

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY env var not set.")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(args.model)

PROMPT = (
  "Write a short, factual paragraph (2–4 sentences) in Wikipedia style that could answer the question. "
  "Use real entity names and concrete facts. Include the specific city if relevant. "
  "Do NOT restate the question. Do NOT invent facts.\n\n"
  "Question: {q}\nPassage:"
)

def _extract_text(resp) -> str:
    # 1) quick accessor
    try:
        t = resp.text
        if t:
            return t.strip()
    except Exception:
        pass
    # 2) candidates/parts
    try:
        for c in (getattr(resp, "candidates", None) or []):
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                texts = []
                for p in parts:
                    pt = getattr(p, "text", None)
                    if pt:
                        texts.append(pt)
                if texts:
                    return " ".join(texts).strip()
    except Exception:
        pass
    return ""

def gen_hypo(question: str, retries: int, timeout: float) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(
                PROMPT.format(q=question),
                generation_config={
                    "max_output_tokens": args.max_tokens,
                    "temperature": args.temp,
                    "top_p": args.top_p,
                },
                request_options={"timeout": timeout},  # <- hard timeout per call
            )
            txt = _extract_text(resp)
            if txt:
                return txt
        except Exception as e:
            if attempt == retries:
                print(f"\n[error] last attempt failed: {e}", flush=True)
        time.sleep(1.0 * attempt)  # backoff
    return ""

def main():
    # Load / init cache
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run: python -m scripts.01_prepare_data")

    # Load queries
    qs = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            qid = ex.get("_id") or ex.get("id") or f"q{i}"
            qs.append((qid, ex["question"]))
    if args.limit > 0:
        qs = qs[:args.limit]

    total = len(qs)
    print(f"Prewarming Gemini hypos | model={args.model} | n={args.n} | queries={total}")
    print(f"Cache: {cache_path}")
    print("-" * 60, flush=True)

    written = skipped = 0
    start = time.time()

    for idx, (qid, question) in enumerate(qs, 1):
        key = f"{qid}::n={args.n}"
        if key in cache:
            print(f"[{idx}/{total}] {qid}: cached — skip", flush=True)
            continue

        print(f"[{idx}/{total}] {qid}: generating ", end="", flush=True)
        hypos = []
        for j in range(args.n):
            h = gen_hypo(question, retries=args.retries, timeout=args.timeout)
            if h:
                hypos.append(h)
                print(".", end="", flush=True)
            else:
                print("x", end="", flush=True)

        if hypos:
            cache[key] = hypos
            written += 1
            print(" ✓", flush=True)
        else:
            skipped += 1
            print(" ✗ (empty)", flush=True)

        # Persist incrementally every few items to be robust to interrupts
        if (written + skipped) % 5 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final write
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    dur = time.time() - start
    print("-" * 60)
    print(f"Done. Added {written}, skipped {skipped}. Elapsed: {dur:.1f}s")
    print(f"Cache at: {cache_path}")

if __name__ == "__main__":
    main()
