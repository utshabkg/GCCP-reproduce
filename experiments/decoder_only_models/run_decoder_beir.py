#!/usr/bin/env python3
"""
Run RG-YN + GCCP + PAGC with a decoder-only LLM on a BEIR-E5 dataset.

Reuses the BEIR-E5 retrieval (data/beir_e5_<dataset>.json) so the
first-stage is fixed; reranks with a decoder-only model. Output
mirrors the Flan-T5-XL BEIR-E5 layout for downstream stat tests.

Usage:
    python experiments/decoder_only_models/run_decoder_beir.py \\
        --dataset scifact \\
        --model /media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ \\
        --short_name qwen2.5-72b-awq
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

from experiments.decoder_only_models.decoder_rankers import (
    DecoderOnlyGCCPRanker,
    DecoderOnlyRGYNRanker,
)
from src.pagc.aggregation import linear_aggregation


def load_data(dataset: str):
    queries = json.loads((REPO_ROOT / f"data/beir_{dataset}_queries.json").read_text())
    qrels = json.loads((REPO_ROOT / f"data/beir_{dataset}_qrels.json").read_text())
    raw = json.loads((REPO_ROOT / f"data/beir_e5_{dataset}.json").read_text())
    e5 = {
        qid: [
            {"docid": p["pid"], "score": p.get("score", 100 - i), "contents": p["text"]}
            for i, p in enumerate(d["passages"])
        ]
        for qid, d in raw.items()
    }
    return queries, qrels, e5


def avg(d, m):
    return sum(r[m] for r in d.values()) / len(d) if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--short_name", required=True)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--max_doc_length", type=int, default=128)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "beir" / args.dataset / f"{args.short_name}_e5"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] decoder-only BEIR-E5 run: {args.model} on {args.dataset}")
    print(f"output: {out_dir}")

    queries, qrels, e5 = load_data(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    print(f"loading {args.model} ...")
    rg_yn = DecoderOnlyRGYNRanker(args.model, max_doc_length=args.max_doc_length)
    gccp = DecoderOnlyGCCPRanker(
        args.model,
        max_doc_length=args.max_doc_length,
        m=10,
        z=10,
        threshold=0.2,
        shared=(rg_yn.model, rg_yn.tokenizer),
    )

    rg_results = {}
    gccp_results = {}
    for qid in tqdm(qids, desc="queries"):
        docs = e5[qid][:100]
        rg_results[qid] = {d: s for d, s in rg_yn.rank(queries[qid], docs)}
        rankings, _ = gccp.rank(queries[qid], docs)
        gccp_results[qid] = {d: s for d, s in rankings}

    pagc_results = {q: linear_aggregation([rg_results[q], gccp_results[q]]) for q in qids}

    e5_run = {q: {d["docid"]: d["score"] for d in e5[q][:100]} for q in qids}
    test_qrels = {q: qrels[q] for q in qids if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut_10", "P_10", "recall_10"})

    ev_runs = {
        "e5": ev.evaluate(e5_run),
        "rg_yn": ev.evaluate(rg_results),
        "gccp": ev.evaluate(gccp_results),
        "pagc": ev.evaluate(pagc_results),
    }

    print(f"\n{'method':<10} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}")
    for name, d in ev_runs.items():
        print(
            f"{name:<10} {avg(d, 'ndcg_cut_10'):>10.4f} "
            f"{avg(d, 'P_10'):>10.4f} {avg(d, 'recall_10'):>10.4f}"
        )

    elapsed = datetime.now() - start
    metrics = {
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "short_name": args.short_name,
            "num_queries": len(qids),
            "elapsed": str(elapsed),
            "timestamp": start.isoformat(),
            "max_doc_length": args.max_doc_length,
            "first_stage": "e5",
        },
        "results": {
            name: {
                "ndcg@10": avg(d, "ndcg_cut_10"),
                "p@10": avg(d, "P_10"),
                "recall@10": avg(d, "recall_10"),
            }
            for name, d in ev_runs.items()
        },
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "rg_yn_scores.json").write_text(json.dumps(rg_results, indent=2))
    (out_dir / "gccp_scores.json").write_text(json.dumps(gccp_results, indent=2))
    (out_dir / "pagc_scores.json").write_text(json.dumps(pagc_results, indent=2))
    print(f"\nDone in {elapsed}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
