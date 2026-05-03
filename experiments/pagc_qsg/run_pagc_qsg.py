#!/usr/bin/env python3
"""
PAGC-RS-YN-GCCP runner: extends our PAGC = (RG-YN + GCCP) to the paper's
PAGC-RS-YN-GCCP variant (the homogeneous three-component aggregation in
Table 4 of Long et al. 2025). Reuses already-saved RG-YN and GCCP
per-query scores; only RG-S is computed fresh here.

(QG is a separate inference pattern -- conditional log-likelihood of the
query given the doc -- and is left for future work; the paper's strongest
variant PAGC-QSG = QG+RG-S+GCCP would build on this script.)

Usage:
    python experiments/pagc_qsg/run_pagc_qsg.py \
        --dataset dl19 --model flan-t5-large \
        --rg_yn_scores results/trec-dl/dl19/flan-t5-large_bm25/rg_yn_scores.json \
        --gccp_scores  results/trec-dl/dl19/flan-t5-large_bm25/gccp_scores.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytrec_eval
from tqdm import tqdm

from experiments.pagc_qsg.rg_s_ranker import RGSRanker
from src.pagc.aggregation import linear_aggregation

MODEL_MAP = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-ul2": "google/flan-ul2",
}


def load_data(dataset: str):
    queries = json.loads((REPO_ROOT / f"data/{dataset}_queries.json").read_text())
    qrels = json.loads((REPO_ROOT / f"data/{dataset}_qrels.json").read_text())
    raw = json.loads((REPO_ROOT / f"data/{dataset}_pyserini_bm25.json").read_text())
    bm25 = {
        qid: [
            {"docid": p["pid"], "score": 100 - i, "contents": p["text"]}
            for i, p in enumerate(d["passages"])
        ]
        for qid, d in raw.items()
    }
    return queries, qrels, bm25


def avg(d, m):
    return sum(r[m] for r in d.values()) / len(d) if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dl19", "dl20"])
    parser.add_argument("--model", default="flan-t5-large", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--rg_yn_scores", type=Path, required=True)
    parser.add_argument("--gccp_scores", type=Path, required=True)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--scale_k", type=int, default=4)
    args = parser.parse_args()

    full_model = MODEL_MAP[args.model]
    out_dir = REPO_ROOT / "results" / "trec-dl" / args.dataset / f"{args.model}_bm25_rs"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] PAGC-RS-YN-GCCP: {args.dataset} / {args.model}")

    queries, qrels, bm25 = load_data(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    rg_yn = json.loads(Path(args.rg_yn_scores).read_text())
    gccp = json.loads(Path(args.gccp_scores).read_text())
    qids = [q for q in qids if q in rg_yn and q in gccp]
    print(f"queries with rg_yn+gccp scores: {len(qids)}")

    # Compute RG-S
    print(f"loading {full_model} for RG-S(0,{args.scale_k})")
    rgs = RGSRanker(full_model, max_doc_length=128, scale_k=args.scale_k)

    rgs_scores = {}
    for qid in tqdm(qids, desc="RG-S queries"):
        docs = bm25[qid][:100]
        rgs_scores[qid] = {d: s for d, s in rgs.rank(queries[qid], docs)}

    # Aggregate
    pagc2 = {q: linear_aggregation([rg_yn[q], gccp[q]]) for q in qids}
    pagc_rs_yn = {q: linear_aggregation([rg_yn[q], rgs_scores[q]]) for q in qids}
    pagc_rs_gccp = {q: linear_aggregation([rgs_scores[q], gccp[q]]) for q in qids}
    pagc_rs_yn_gccp = {q: linear_aggregation([rgs_scores[q], rg_yn[q], gccp[q]]) for q in qids}

    test_qrels = {q: qrels[q] for q in qids if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut_10", "P_10", "recall_10"})

    runs = {
        "rg_yn":          ev.evaluate(rg_yn),
        "rg_s":           ev.evaluate(rgs_scores),
        "gccp":           ev.evaluate(gccp),
        "pagc (yn+gccp)": ev.evaluate(pagc2),
        "pagc-rs-yn":     ev.evaluate(pagc_rs_yn),
        "pagc-rs-gccp":   ev.evaluate(pagc_rs_gccp),
        "pagc-rs-yn-gccp": ev.evaluate(pagc_rs_yn_gccp),
    }

    print(f"\n{'method':<20} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}")
    for name, d in runs.items():
        print(
            f"{name:<20} {avg(d, 'ndcg_cut_10'):>10.4f} "
            f"{avg(d, 'P_10'):>10.4f} {avg(d, 'recall_10'):>10.4f}"
        )

    elapsed = datetime.now() - start
    metrics = {
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "scale_k": args.scale_k,
            "num_queries": len(qids),
            "elapsed": str(elapsed),
            "timestamp": start.isoformat(),
        },
        "results": {
            name: {
                "ndcg@10": avg(d, "ndcg_cut_10"),
                "p@10": avg(d, "P_10"),
                "recall@10": avg(d, "recall_10"),
            }
            for name, d in runs.items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "rg_s_scores.json").write_text(json.dumps(rgs_scores))
    (out_dir / "pagc_rs_yn_gccp_scores.json").write_text(json.dumps(pagc_rs_yn_gccp))

    print(f"\nDone in {elapsed}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
