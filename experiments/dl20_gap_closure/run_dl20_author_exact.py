#!/usr/bin/env python3
"""
DL20 T5-Large gap closure: rerun the GCCP pipeline using anchors generated
by the author's exact MDS algorithm (NLTK + 200/128 hybrid + abs-Fiedler
minority-side selection), keeping everything else identical.

Compares to results/trec-dl/dl20/flan-t5-large_bm25/ (our re-run with
the in-tree spectral MDS) to isolate the contribution of segmentation
differences to the residual ~5.7% gap.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytrec_eval
from tqdm import tqdm

from experiments.dl20_gap_closure.author_exact_mds import author_exact_anchor
from src.gccp.gccp_ranker import GCCPRanker
from src.pagc.aggregation import linear_aggregation
from src.pointwise.rankers import RGYNRanker


MODEL_NAME = "google/flan-t5-large"
DATASET = "dl20"
OUTPUT_DIR = REPO_ROOT / "results" / "trec-dl" / DATASET / "flan-t5-large_bm25_authorMDS"


def load_data():
    queries = json.loads((REPO_ROOT / f"data/{DATASET}_queries.json").read_text())
    qrels = json.loads((REPO_ROOT / f"data/{DATASET}_qrels.json").read_text())
    raw = json.loads((REPO_ROOT / f"data/{DATASET}_pyserini_bm25.json").read_text())
    bm25 = {
        qid: [
            {"docid": p["pid"], "score": 100 - i, "contents": p["text"]}
            for i, p in enumerate(d["passages"])
        ]
        for qid, d in raw.items()
    }
    return queries, qrels, bm25


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    print(f"[{start:%H:%M:%S}] starting DL20 author-exact MDS rerun")

    queries, qrels, bm25 = load_data()
    qids = list(queries.keys())
    print(f"queries: {len(qids)}")

    print(f"loading {MODEL_NAME}")
    rg_yn = RGYNRanker(MODEL_NAME, template_idx=0, target_tokens=("yes", "no"), max_doc_length=128)
    gccp = GCCPRanker(
        MODEL_NAME, max_doc_length=128, m=10, z=10, threshold=0.2, use_spacy=False
    )

    rg_results = {}
    gccp_results = {}
    for qid in tqdm(qids, desc="queries"):
        docs = bm25[qid][:100]
        query = queries[qid]

        rg_results[qid] = {d: s for d, s in rg_yn.rank(query, docs)}

        anchor = author_exact_anchor(docs, m=10, z=10, threshold=0.2)
        rankings, _ = gccp.rank(query, docs, anchor=anchor)
        gccp_results[qid] = {d: s for d, s in rankings}

    pagc_results = {q: linear_aggregation([rg_results[q], gccp_results[q]]) for q in qids}

    bm25_run = {q: {d["docid"]: d["score"] for d in bm25[q][:100]} for q in qids}
    test_qrels = {q: qrels[q] for q in qids if q in qrels}
    ev = pytrec_eval.RelevanceEvaluator(test_qrels, {"ndcg_cut_10", "P_10", "recall_10"})

    def avg(d, m):
        return sum(r[m] for r in d.values()) / len(d)

    ev_runs = {
        "bm25": ev.evaluate(bm25_run),
        "rg_yn": ev.evaluate(rg_results),
        "gccp": ev.evaluate(gccp_results),
        "pagc": ev.evaluate(pagc_results),
    }

    print(f"\n{'method':<10} {'NDCG@10':>10} {'P@10':>10} {'R@10':>10}")
    for name, eval_d in ev_runs.items():
        print(f"{name:<10} {avg(eval_d, 'ndcg_cut_10'):>10.4f} "
              f"{avg(eval_d, 'P_10'):>10.4f} {avg(eval_d, 'recall_10'):>10.4f}")

    elapsed = datetime.now() - start
    metrics = {
        "experiment": {
            "dataset": DATASET,
            "model": "flan-t5-large",
            "anchor": "author_exact_mds (NLTK + 200/128 hybrid + abs-Fiedler minority)",
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
            for name, d in ev_runs.items()
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "rg_yn_scores.json").write_text(json.dumps(rg_results, indent=2))
    (OUTPUT_DIR / "gccp_scores.json").write_text(json.dumps(gccp_results, indent=2))
    (OUTPUT_DIR / "pagc_scores.json").write_text(json.dumps(pagc_results, indent=2))

    # Comparison with in-tree-MDS baseline
    base = REPO_ROOT / "results/trec-dl/dl20/flan-t5-large_bm25/metrics.json"
    if base.exists():
        baseline = json.loads(base.read_text())["results"]
        print(f"\n{'method':<10} {'in-tree MDS':>14} {'author MDS':>14} {'delta':>10}")
        for name in ["bm25", "rg_yn", "gccp", "pagc"]:
            old = baseline.get(name, {}).get("ndcg@10", float("nan"))
            new = metrics["results"][name]["ndcg@10"]
            print(f"{name:<10} {old:>14.4f} {new:>14.4f} {new - old:>+10.4f}")

    print(f"\nDone in {elapsed}. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
