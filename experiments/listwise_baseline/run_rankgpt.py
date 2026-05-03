#!/usr/bin/env python3
"""
Listwise (RankGPT-style) reranking baseline using llm-rankers, for fair
comparison with PAGC at the same model scale.

We follow the original RankGPT setup: sliding window over candidates,
window=20, step=10. We use the same Flan-T5 backbones the original
GCCP paper compared against in their Table 2.

Usage:
    python experiments/listwise_baseline/run_rankgpt.py \
        --dataset dl19 --model flan-t5-large
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
from llmrankers.listwise import ListwiseLlmRanker
from llmrankers.rankers import SearchResult

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
    parser.add_argument(
        "--model", default="flan-t5-large", choices=list(MODEL_MAP.keys())
    )
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--step", type=int, default=10)
    args = parser.parse_args()

    full_model = MODEL_MAP[args.model]
    out_dir = REPO_ROOT / "results" / "trec-dl" / args.dataset / f"{args.model}_rankgpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] RankGPT-listwise: {args.dataset} / {args.model}")

    queries, qrels, bm25 = load_data(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}, window={args.window}, step={args.step}")

    print(f"loading {full_model}")
    ranker = ListwiseLlmRanker(
        model_name_or_path=full_model,
        tokenizer_name_or_path=full_model,
        device="cuda",
        window_size=args.window,
        step_size=args.step,
        scoring="generation",
    )

    rankings = {}
    for qid in tqdm(qids, desc="queries"):
        sr_in = [
            SearchResult(docid=d["docid"], score=d["score"], text=d["contents"])
            for d in bm25[qid][:100]
        ]
        sr_out = ranker.rerank(queries[qid], sr_in)
        # Score via descending position so pytrec_eval still works
        rankings[qid] = {sr.docid: float(len(sr_out) - i) for i, sr in enumerate(sr_out)}

    test_qrels = {q: qrels[q] for q in qids if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut_10", "P_10", "recall_10"})
    bm25_run = {q: {d["docid"]: d["score"] for d in bm25[q][:100]} for q in qids}

    runs = {"bm25": ev.evaluate(bm25_run), "rankgpt": ev.evaluate(rankings)}
    print(f"\n{'method':<12} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}")
    for name, d in runs.items():
        print(
            f"{name:<12} {avg(d, 'ndcg_cut_10'):>10.4f} "
            f"{avg(d, 'P_10'):>10.4f} {avg(d, 'recall_10'):>10.4f}"
        )

    elapsed = datetime.now() - start
    metrics = {
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "window": args.window,
            "step": args.step,
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
    (out_dir / "rankgpt_rankings.json").write_text(json.dumps(rankings))
    print(f"\nDone in {elapsed}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
