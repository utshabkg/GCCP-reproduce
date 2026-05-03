#!/usr/bin/env python3
"""
Cross-model significance test: Qwen-2.5-7B-Instruct vs Flan-T5-XL on
DL19 PAGC. Both rerank the same BM25 top-100 candidates so per-query
NDCG@10 deltas are well-defined.

Run from gccp-reproduce env (uses pytrec_eval; reads any saved scores).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pytrec_eval

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def per_q(qrels, scores, metric="ndcg_cut_10"):
    common = {q: qrels[q] for q in scores if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(common, {metric})
    return {q: r[metric] for q, r in ev.evaluate({q: scores[q] for q in common}).items()}


def bootstrap(base, sysrun, n=10000, seed=929):
    rng = np.random.default_rng(seed)
    diffs = sysrun - base
    delta = float(diffs.mean())
    idx = rng.integers(0, base.size, size=(n, base.size))
    bs = diffs[idx].mean(axis=1)
    p = 2.0 * (np.mean(bs <= 0) if delta >= 0 else np.mean(bs >= 0))
    p = min(1.0, float(p))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return delta, p, float(lo), float(hi)


def main():
    qrels = json.load(open(REPO / "data/dl19_qrels.json"))

    # Both PAGC scores are saved
    qwen = json.load(open(REPO / "results/trec-dl/dl19/qwen2.5-7b_bm25/pagc_scores.json"))
    # T5-XL PAGC is constructed by linear_aggregation of saved RG-YN+GCCP at run time;
    # we don't have it as a single file, so reconstruct it deterministically from saved scores.
    # The T5-XL/BM25 result was ran with the original pipeline and reported NDCG@10=0.7030.
    # Saved scores live under results/trec-dl/dl19/flan-t5-xl_e5 (E5 not BM25).
    # We need the T5-XL/BM25 saved per-query scores; check if available.
    cand = REPO / "results/trec-dl/dl19/flan-t5-xl_bm25/pagc_scores.json"
    if not cand.exists():
        # Fall back to reading legacy flat-bm25 numbers if present, else bail with a clear note.
        print("No flan-t5-xl_bm25/pagc_scores.json found — T5-XL/BM25 was run before per-query saving was added.")
        print("Reporting Qwen-2.5-7B PAGC's per-query distribution + 95% CI on its mean instead.")
        q_pq = per_q(qrels, qwen)
        arr = np.array(list(q_pq.values()))
        print(f"Qwen-2.5-7B PAGC NDCG@10  mean={arr.mean():.4f}  std={arr.std():.4f}  n={arr.size}")
        # Bootstrap one-sample CI on the mean
        rng = np.random.default_rng(929)
        idx = rng.integers(0, arr.size, size=(10000, arr.size))
        means = arr[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        print(f"  95% CI on mean Qwen PAGC NDCG@10: [{lo:.4f}, {hi:.4f}]")
        # Compare against the published T5-XL value 0.7030 as a constant offset:
        # Bootstrap distribution > 0.7030 ?
        n_above = int((means > 0.7030).sum())
        print(f"  Bootstrap resamples in which Qwen mean > 0.7030 (T5-XL): "
              f"{n_above}/10000 = {n_above/10000:.4f}")
        # One-sided p-value (Qwen > T5-XL constant) is 1 - that fraction
        # but this is an approximation: it ignores the variance in T5-XL
        # since we don't have its per-query scores. Note caveat in the paper.
        return

    t5xl = json.load(open(cand))
    qwen_pq = per_q(qrels, qwen)
    t5xl_pq = per_q(qrels, t5xl)
    common = sorted(set(qwen_pq) & set(t5xl_pq))
    a = np.array([qwen_pq[q] for q in common])
    b = np.array([t5xl_pq[q] for q in common])
    delta, p, lo, hi = bootstrap(b, a)
    print(f"DL19 PAGC: Qwen-2.5-7B vs Flan-T5-XL (n={len(common)})")
    print(f"  Qwen mean = {a.mean():.4f}")
    print(f"  T5-XL mean = {b.mean():.4f}")
    print(f"  delta = {delta:+.4f}, p (two-sided, 10k bootstrap) = {p:.4f}")
    print(f"  95% CI on delta: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
