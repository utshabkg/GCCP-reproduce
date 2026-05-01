#!/usr/bin/env python3
"""
Paired bootstrap significance tests for ranking experiments.

Given two saved score files (e.g., RG-YN vs PAGC), reports per-query NDCG@10
deltas and a paired bootstrap p-value over the difference of means.

Usage:
    python experiments/statistical_tests/paired_bootstrap.py \
        --baseline_scores results/trec-dl/dl19/flan-t5-xl_e5/rg_yn_scores.json \
        --system_scores   results/trec-dl/dl19/flan-t5-xl_e5/pagc_scores.json \
        --qrels           data/dl19_qrels.json \
        --metric          ndcg_cut_10 \
        --resamples       1000

Single-system mode (compare against a per-query metric file or against zero):
    python experiments/statistical_tests/paired_bootstrap.py \
        --baseline_scores ... --system_scores ... --qrels ... \
        --output results/stat_tests/dl19_t5xl_e5_pagc_vs_rg_yn.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytrec_eval


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(Path(path).read_text())


def per_query_metric(
    qrels: Dict[str, Dict[str, int]],
    scores: Dict[str, Dict[str, float]],
    metric: str,
) -> Dict[str, float]:
    common = {qid: qrels[qid] for qid in scores if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(common, {metric})
    per_q = evaluator.evaluate({qid: scores[qid] for qid in common})
    return {qid: r[metric] for qid, r in per_q.items()}


def paired_bootstrap(
    baseline: np.ndarray,
    system: np.ndarray,
    resamples: int = 1000,
    seed: int = 929,
) -> Tuple[float, float, float, float]:
    """Two-sided paired bootstrap on the mean of (system - baseline).

    Returns:
        delta_mean   : mean(system - baseline) on the original sample
        p_two_sided  : two-sided p-value (resamples in which sign flips)
        ci_lo, ci_hi : 95% CI on the bootstrap distribution of the mean delta
    """
    assert baseline.shape == system.shape
    n = baseline.size
    rng = np.random.default_rng(seed)
    diffs = system - baseline
    delta_mean = float(diffs.mean())

    # Bootstrap distribution of the mean delta
    idx = rng.integers(0, n, size=(resamples, n))
    bs = diffs[idx].mean(axis=1)

    # Two-sided p-value: how often does the resampled mean cross zero?
    if delta_mean >= 0:
        p_two_sided = 2.0 * float(np.mean(bs <= 0))
    else:
        p_two_sided = 2.0 * float(np.mean(bs >= 0))
    p_two_sided = min(1.0, p_two_sided)

    ci_lo, ci_hi = np.percentile(bs, [2.5, 97.5])
    return delta_mean, p_two_sided, float(ci_lo), float(ci_hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_scores", required=True, type=Path)
    parser.add_argument("--system_scores", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument(
        "--metric",
        default="ndcg_cut_10",
        choices=["ndcg_cut_10", "P_10", "recall_10", "ndcg_cut_100"],
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--label",
        default="",
        help="Free-form label, e.g. 'DL19 / Flan-T5-XL / E5: PAGC vs RG-YN'",
    )
    args = parser.parse_args()

    baseline = load_scores(args.baseline_scores)
    system = load_scores(args.system_scores)
    qrels = load_scores(args.qrels)

    base_pq = per_query_metric(qrels, baseline, args.metric)
    sys_pq = per_query_metric(qrels, system, args.metric)
    common = sorted(set(base_pq) & set(sys_pq))

    base_arr = np.array([base_pq[q] for q in common])
    sys_arr = np.array([sys_pq[q] for q in common])

    delta_mean, p, ci_lo, ci_hi = paired_bootstrap(
        base_arr, sys_arr, args.resamples, args.seed
    )

    n_better = int((sys_arr > base_arr).sum())
    n_worse = int((sys_arr < base_arr).sum())
    n_tied = len(common) - n_better - n_worse

    print(f"Label:    {args.label}")
    print(f"Metric:   {args.metric}")
    print(f"Queries:  {len(common)} (baseline={len(base_pq)}, system={len(sys_pq)})")
    print(f"Baseline mean: {base_arr.mean():.4f}")
    print(f"System mean:   {sys_arr.mean():.4f}")
    print(f"Delta (sys - base): {delta_mean:+.4f}")
    print(f"Per-query: {n_better} better / {n_worse} worse / {n_tied} tied")
    print(f"Paired bootstrap (n={args.resamples}, seed={args.seed}):")
    print(f"  95% CI on mean delta: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Two-sided p-value:    {p:.4f}")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  Significance:         {sig}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "metric": args.metric,
                    "baseline_scores": str(args.baseline_scores),
                    "system_scores": str(args.system_scores),
                    "qrels": str(args.qrels),
                    "n_queries": len(common),
                    "baseline_mean": float(base_arr.mean()),
                    "system_mean": float(sys_arr.mean()),
                    "delta_mean": delta_mean,
                    "p_value_two_sided": p,
                    "ci_95_lo": ci_lo,
                    "ci_95_hi": ci_hi,
                    "n_better": n_better,
                    "n_worse": n_worse,
                    "n_tied": n_tied,
                    "resamples": args.resamples,
                    "seed": args.seed,
                    "significance": sig,
                },
                indent=2,
            )
        )
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
