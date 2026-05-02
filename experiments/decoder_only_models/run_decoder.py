#!/usr/bin/env python3
"""
Run RG-YN + GCCP + PAGC with a decoder-only LLM on TREC DL 2019/2020.

Saves per-query scores under
results/trec-dl/dl{19,20}/<short_model_name>/{rg_yn,gccp,pagc}_scores.json
and a metrics.json compatible with the rest of the pipeline.

Usage:
    python experiments/decoder_only_models/run_decoder.py \
        --dataset dl19 \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --short_name llama3-8b
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
    return sum(r[m] for r in d.values()) / len(d)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["dl19", "dl20"])
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--short_name",
        required=True,
        help="Short label used in output dir (e.g. llama3-8b, qwen2.5-7b)",
    )
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--max_doc_length", type=int, default=128)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "trec-dl" / args.dataset / f"{args.short_name}_bm25"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] decoder-only run: {args.model} on {args.dataset}")
    print(f"output: {out_dir}")

    queries, qrels, bm25 = load_data(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    print(f"loading {args.model} ...")
    rg_yn = DecoderOnlyRGYNRanker(args.model, max_doc_length=args.max_doc_length)
    gccp = DecoderOnlyGCCPRanker(
        args.model, max_doc_length=args.max_doc_length, m=10, z=10, threshold=0.2
    )

    rg_results = {}
    gccp_results = {}
    for qid in tqdm(qids, desc="queries"):
        docs = bm25[qid][:100]
        rg_results[qid] = {d: s for d, s in rg_yn.rank(queries[qid], docs)}
        rankings, _ = gccp.rank(queries[qid], docs)
        gccp_results[qid] = {d: s for d, s in rankings}

    pagc_results = {q: linear_aggregation([rg_results[q], gccp_results[q]]) for q in qids}

    bm25_run = {q: {d["docid"]: d["score"] for d in bm25[q][:100]} for q in qids}
    test_qrels = {q: qrels[q] for q in qids if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut_10", "P_10", "recall_10"})

    ev_runs = {
        "bm25": ev.evaluate(bm25_run),
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
