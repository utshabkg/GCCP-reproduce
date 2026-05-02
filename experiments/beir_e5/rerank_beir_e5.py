#!/usr/bin/env python3
"""
Run RG-YN + GCCP + PAGC on BEIR E5 retrieval results (data/beir_e5_<dataset>.json).

Run in the gccp-reproduce env (transformers 4.36 + Flan-T5-XL).

Usage:
    python experiments/beir_e5/rerank_beir_e5.py --dataset scifact --model flan-t5-xl
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

MODEL_MAP = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-ul2": "google/flan-ul2",
}


def load_data(dataset: str):
    queries = json.loads(
        (REPO_ROOT / f"data/beir_{dataset}_queries.json").read_text()
    )
    qrels = json.loads(
        (REPO_ROOT / f"data/beir_{dataset}_qrels.json").read_text()
    )
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
    parser.add_argument(
        "--model",
        default="flan-t5-xl",
        choices=["flan-t5-large", "flan-t5-xl", "flan-ul2"],
    )
    parser.add_argument("--num_queries", type=int, default=None)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "beir" / args.dataset / f"{args.model}_e5"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_model = MODEL_MAP[args.model]
    start = datetime.now()
    print(f"[{start:%H:%M:%S}] BEIR-E5 rerank: {args.dataset} / {args.model}")

    queries, qrels, e5 = load_data(args.dataset)
    qids = sorted(set(queries) & set(e5))
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    print(f"loading {full_model}")
    from src.gccp.gccp_ranker import GCCPRanker
    from src.pagc.aggregation import linear_aggregation
    from src.pointwise.rankers import RGYNRanker

    rg_yn = RGYNRanker(
        full_model, template_idx=0, target_tokens=("yes", "no"), max_doc_length=128
    )
    gccp = GCCPRanker(
        full_model, max_doc_length=128, m=10, z=10, threshold=0.2, use_spacy=False
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

    runs = {
        "e5": ev.evaluate(e5_run),
        "rg_yn": ev.evaluate(rg_results),
        "gccp": ev.evaluate(gccp_results),
        "pagc": ev.evaluate(pagc_results),
    }
    print(f"\n{'method':<10} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}")
    for name, d in runs.items():
        print(
            f"{name:<10} {avg(d, 'ndcg_cut_10'):>10.4f} "
            f"{avg(d, 'P_10'):>10.4f} {avg(d, 'recall_10'):>10.4f}"
        )

    elapsed = datetime.now() - start
    metrics = {
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "retrieval": "e5-base-v2",
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
    (out_dir / "rg_yn_scores.json").write_text(json.dumps(rg_results))
    (out_dir / "gccp_scores.json").write_text(json.dumps(gccp_results))
    (out_dir / "pagc_scores.json").write_text(json.dumps(pagc_results))
    print(f"\nDone in {elapsed}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
