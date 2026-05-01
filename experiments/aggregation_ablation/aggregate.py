#!/usr/bin/env python3
"""
Aggregation method ablation for PAGC.

Loads pre-saved per-query RG-YN and GCCP scores and compares aggregation
strategies: linear (paper default), Borda, Condorcet, Copeland, and an
alpha-weighted linear sweep.

Usage:
    python experiments/aggregation_ablation/aggregate.py \
        --rg_yn_scores results/trec-dl/dl19/flan-t5-xl_e5/rg_yn_scores.json \
        --gccp_scores  results/trec-dl/dl19/flan-t5-xl_e5/gccp_scores.json \
        --qrels        data/dl19_qrels.json \
        --output       results/ablations/aggregation_dl19_t5xl_e5.json \
        --label        "DL19 / Flan-T5-XL / E5"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytrec_eval

from src.pagc.aggregation import (
    borda_aggregation,
    condorcet_aggregation,
    copeland_aggregation,
    linear_aggregation,
)


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(Path(path).read_text())


def scores_to_ranking(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def alpha_weighted_linear(
    rg_yn: Dict[str, float], gccp: Dict[str, float], alpha: float
) -> Dict[str, float]:
    """alpha * RG-YN + (1-alpha) * GCCP (both min-max normalized to [0,1])."""

    def normalize(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        vmin, vmax = min(d.values()), max(d.values())
        if vmax - vmin == 0:
            return {k: 0.5 for k in d}
        return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

    rg_n, gc_n = normalize(rg_yn), normalize(gccp)
    docids = set(rg_n) | set(gc_n)
    return {
        d: alpha * rg_n.get(d, 0.0) + (1.0 - alpha) * gc_n.get(d, 0.0) for d in docids
    }


def evaluate_run(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    common = {qid: qrels[qid] for qid in run if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(
        common, {"ndcg_cut_10", "P_10", "recall_10"}
    )
    per_q = evaluator.evaluate({qid: run[qid] for qid in common})
    if not per_q:
        return {"ndcg@10": 0.0, "p@10": 0.0, "recall@10": 0.0, "n_eval": 0}
    n = len(per_q)
    return {
        "ndcg@10": sum(r["ndcg_cut_10"] for r in per_q.values()) / n,
        "p@10": sum(r["P_10"] for r in per_q.values()) / n,
        "recall@10": sum(r["recall_10"] for r in per_q.values()) / n,
        "n_eval": n,
    }


def aggregate_all(
    rg_yn: Dict[str, Dict[str, float]],
    gccp: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    alphas: List[float],
) -> Dict[str, Dict]:
    qids = sorted(set(rg_yn) & set(gccp))
    out: Dict[str, Dict] = {}

    # Single-method baselines
    out["rg_yn_only"] = evaluate_run(qrels, {q: rg_yn[q] for q in qids})
    out["gccp_only"] = evaluate_run(qrels, {q: gccp[q] for q in qids})

    # Linear (paper default, equal weight)
    linear_run = {q: linear_aggregation([rg_yn[q], gccp[q]]) for q in qids}
    out["linear (paper, alpha=0.5)"] = evaluate_run(qrels, linear_run)

    # Alpha-weighted linear sweep
    for alpha in alphas:
        run = {q: alpha_weighted_linear(rg_yn[q], gccp[q], alpha) for q in qids}
        out[f"linear (alpha={alpha:.2f})"] = evaluate_run(qrels, run)

    # Rank-based methods
    borda_run = {
        q: borda_aggregation(
            [scores_to_ranking(rg_yn[q]), scores_to_ranking(gccp[q])]
        )
        for q in qids
    }
    out["borda"] = evaluate_run(qrels, borda_run)

    condorcet_run = {
        q: condorcet_aggregation(
            [scores_to_ranking(rg_yn[q]), scores_to_ranking(gccp[q])]
        )
        for q in qids
    }
    out["condorcet"] = evaluate_run(qrels, condorcet_run)

    copeland_run = {
        q: copeland_aggregation(
            [scores_to_ranking(rg_yn[q]), scores_to_ranking(gccp[q])]
        )
        for q in qids
    }
    out["copeland"] = evaluate_run(qrels, copeland_run)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rg_yn_scores", required=True, type=Path)
    parser.add_argument("--gccp_scores", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", default="", help="Free-form label for this config")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Alphas for alpha-weighted linear (alpha * RG-YN + (1-alpha) * GCCP)",
    )
    args = parser.parse_args()

    rg_yn = load_scores(args.rg_yn_scores)
    gccp = load_scores(args.gccp_scores)
    qrels = load_scores(args.qrels)

    print(f"Label:           {args.label}")
    print(f"RG-YN scores:    {args.rg_yn_scores}  ({len(rg_yn)} queries)")
    print(f"GCCP scores:     {args.gccp_scores}   ({len(gccp)} queries)")
    print(f"Qrels:           {args.qrels}        ({len(qrels)} queries)")

    methods = aggregate_all(rg_yn, gccp, qrels, args.alphas)

    print(f"\n{'Method':<32} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}  N")
    print("-" * 70)
    for name, m in methods.items():
        print(
            f"{name:<32} {m['ndcg@10']:>10.4f} {m['p@10']:>10.4f} "
            f"{m['recall@10']:>10.4f}  {m['n_eval']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "label": args.label,
                "rg_yn_scores": str(args.rg_yn_scores),
                "gccp_scores": str(args.gccp_scores),
                "qrels": str(args.qrels),
                "alphas": args.alphas,
                "methods": methods,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
