#!/usr/bin/env python3
"""
Per-query latency measurement for RG-YN, GCCP, and PAGC.

Measures wall-clock seconds to score the top-100 BM25 candidates of N
queries from DL19, after a model warmup pass. Reports mean / std / total
per method, and aggregate per-document throughput. Useful to substantiate
the paper's "pointwise efficiency" claim quantitatively.

Usage:
    python experiments/efficiency_analysis/measure_latency.py \
        --model flan-t5-large --dataset dl19 --num_queries 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

MODEL_MAP = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-ul2": "google/flan-ul2",
}


def load_data(dataset: str):
    queries = json.loads((REPO_ROOT / f"data/{dataset}_queries.json").read_text())
    raw = json.loads((REPO_ROOT / f"data/{dataset}_pyserini_bm25.json").read_text())
    bm25 = {
        qid: [
            {"docid": p["pid"], "score": 100 - i, "contents": p["text"]}
            for i, p in enumerate(d["passages"])
        ]
        for qid, d in raw.items()
    }
    return queries, bm25


def time_block():
    """Return a context-manager that measures wall-clock seconds."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dl19", choices=["dl19", "dl20"])
    parser.add_argument(
        "--model",
        required=True,
        help="Short name (flan-t5-large/-xl, flan-ul2) or full HF id",
    )
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--warmup_queries", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: results/efficiency/<dataset>_<model>.json)",
    )
    args = parser.parse_args()

    full_model = MODEL_MAP.get(args.model, args.model)
    short = args.model
    out_path = args.output or (
        REPO_ROOT / "results" / "efficiency" / f"{args.dataset}_{short}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {full_model} ...")
    from src.gccp.gccp_ranker import GCCPRanker
    from src.pagc.aggregation import linear_aggregation
    from src.pointwise.rankers import RGYNRanker

    rg_yn = RGYNRanker(
        full_model, template_idx=0, target_tokens=("yes", "no"), max_doc_length=128
    )
    gccp = GCCPRanker(
        full_model, max_doc_length=128, m=10, z=10, threshold=0.2, use_spacy=False
    )

    queries, bm25 = load_data(args.dataset)
    qids = list(queries.keys())[: args.num_queries + args.warmup_queries]
    print(f"using {len(qids)} queries ({args.warmup_queries} warmup, {args.num_queries} timed)")

    rg_times, gccp_times, pagc_times = [], [], []

    for i, qid in enumerate(qids):
        is_warmup = i < args.warmup_queries
        docs = bm25[qid][:100]
        query = queries[qid]
        n_docs = len(docs)

        # RG-YN
        t0 = time_block()
        rg_run = rg_yn.rank(query, docs)
        rg_dict = {d: s for d, s in rg_run}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_rg = time.perf_counter() - t0

        # GCCP (anchor + scoring)
        t0 = time_block()
        gccp_run, _ = gccp.rank(query, docs)
        gccp_dict = {d: s for d, s in gccp_run}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gccp = time.perf_counter() - t0

        # PAGC (negligible aggregation cost; included for completeness)
        t0 = time.perf_counter()
        _ = linear_aggregation([rg_dict, gccp_dict])
        t_pagc = time.perf_counter() - t0
        # report PAGC as RG-YN + GCCP + aggregation since pipeline-wise that's the cost
        t_pagc_full = t_rg + t_gccp + t_pagc

        marker = "[warmup]" if is_warmup else "[timed] "
        print(
            f"{marker} qid={qid} ndocs={n_docs}  "
            f"rg={t_rg:.3f}s  gccp={t_gccp:.3f}s  pagc(end-to-end)={t_pagc_full:.3f}s"
        )
        if not is_warmup:
            rg_times.append(t_rg)
            gccp_times.append(t_gccp)
            pagc_times.append(t_pagc_full)

    def summarize(name: str, ts: list[float]) -> dict:
        s = {
            "method": name,
            "n": len(ts),
            "mean_sec_per_query": statistics.mean(ts),
            "stdev_sec_per_query": statistics.stdev(ts) if len(ts) > 1 else 0.0,
            "total_sec": sum(ts),
            "ms_per_doc": 1000.0 * statistics.mean(ts) / 100.0,  # 100 docs/query
        }
        print(
            f"\n{name:<6}  mean={s['mean_sec_per_query']:.3f}s/q  "
            f"std={s['stdev_sec_per_query']:.3f}s  total={s['total_sec']:.1f}s  "
            f"~{s['ms_per_doc']:.1f} ms/doc"
        )
        return s

    summary = {
        "experiment": {
            "dataset": args.dataset,
            "model": full_model,
            "short_name": short,
            "num_queries": args.num_queries,
            "warmup_queries": args.warmup_queries,
            "docs_per_query": 100,
            "max_doc_length": 128,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now().isoformat(),
        },
        "methods": {
            "rg_yn": summarize("RG-YN", rg_times),
            "gccp": summarize("GCCP", gccp_times),
            "pagc": summarize("PAGC", pagc_times),
        },
    }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
