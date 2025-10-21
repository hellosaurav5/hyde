from pathlib import Path
import jsonlines, pandas as pd, numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
runs = root/"runs"
gold_path = runs/"gold.jsonl"

def load(g):  # qid -> [pids]
    d={}
    for rec in jsonlines.open(g): d[rec["qid"]]=rec["relevant_pids"]
    return d

def load_ret(p):  # qid -> [pids]
    d={}
    for rec in jsonlines.open(p): d[rec["qid"]]=rec["pids"]
    return d

def recall_at_k(gold, ret, k=10):
    vals=[]
    for q, rel in gold.items():
        rr = any(pid in set(ret.get(q,[])[:k]) for pid in rel)
        vals.append(int(rr))
    return float(np.mean(vals))

def mrr_at_k(gold, ret, k=10):
    vals=[]
    for q, rel in gold.items():
        rr=0.0
        for r,p in enumerate(ret.get(q,[])[:k],1):
            if p in set(rel): rr=1/r; break
        vals.append(rr)
    return float(np.mean(vals))

gold = load(gold_path)
rows = []
for name, fn in [
    ("Baseline","retrieval_baseline.jsonl"),
    ("HyDE(1)","retrieval_hyde.jsonl"),
    ("Multi-HyDE(2)","retrieval_multi.jsonl"),
    ("Fusion","retrieval_fuse.jsonl"),
    ("Hybrid","retrieval_hybrid.jsonl")
]:
    p = runs/fn
    if p.exists():
        ret = load_ret(p)
        rows.append({"Method":name, "Recall@10":recall_at_k(gold, ret, 10), "MRR@10":mrr_at_k(gold, ret, 10)})
df = pd.DataFrame(rows).round(3).sort_values("Recall@10", ascending=False)
df.to_csv(runs/"report_metrics.csv", index=False)
print(df)

# Plot
plt.figure()
plt.bar(df["Method"], df["Recall@10"])
plt.title("Recall@10")
plt.savefig(runs/"recall10.png", bbox_inches="tight")
plt.figure()
plt.bar(df["Method"], df["MRR@10"])
plt.title("MRR@10")
plt.savefig(runs/"mrr10.png", bbox_inches="tight")
