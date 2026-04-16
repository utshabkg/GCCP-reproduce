#!/usr/bin/env python3
import sys
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytrec_eval
from tqdm import tqdm

from src.pointwise.rankers import RGYNRanker
from src.gccp.gccp_ranker import GCCPRanker
from src.pagc.aggregation import linear_aggregation


MODEL_MAP = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-ul2": "google/flan-ul2",
}


def load_e5_data(dataset: str, retrieval_file: str = None):
    with open(f"data/{dataset}_queries.json", "r") as f:
        queries = json.load(f)

    with open(f"data/{dataset}_qrels.json", "r") as f:
        qrels = json.load(f)

    retrieval_file = retrieval_file or f"data/{dataset}_e5_results.json"
    with open(retrieval_file, "r") as f:
        raw_results = json.load(f)

    e5_results = {}
    for qid, data in raw_results.items():
        e5_results[qid] = [
            {
                "docid": p["pid"],
                "score": p.get("score", 100 - i),
                "contents": p["text"],
            }
            for i, p in enumerate(data["passages"])
        ]

    return queries, qrels, e5_results


def run_experiment(
    dataset: str,
    model_name: str,
    retrieval_file: str = None,
    num_queries: int = None,
    output_dir: str = None,
):
    start_time = datetime.now()

    print("=" * 70)
    print(f"GCCP Experiment with E5: {dataset.upper()} with {model_name}")
    print("=" * 70)
    print(f"Start time: {start_time}")

    queries, qrels, e5_results = load_e5_data(dataset, retrieval_file)

    all_qids = list(queries.keys())
    if num_queries:
        all_qids = all_qids[:num_queries]

    full_model_name = MODEL_MAP.get(model_name, model_name)
    print(f"\nLoading model: {full_model_name}")

    rg_yn_ranker = RGYNRanker(
        full_model_name,
        template_idx=0,
        target_tokens=("yes", "no"),
        max_doc_length=128,
    )

    gccp_ranker = GCCPRanker(
        full_model_name,
        max_doc_length=128,
        m=10,
        z=10,
        threshold=0.2,
        use_spacy=False,
    )

    rg_yn_results = {}
    gccp_results = {}

    for qid in tqdm(all_qids, desc="Processing queries"):
        query = queries[qid]
        docs = e5_results[qid][:100]

        rg_yn_rankings = rg_yn_ranker.rank(query, docs)
        rg_yn_results[qid] = {docid: score for docid, score in rg_yn_rankings}

        gccp_rankings, _ = gccp_ranker.rank(query, docs)
        gccp_results[qid] = {docid: score for docid, score in gccp_rankings}

    pagc_results = {}
    for qid in all_qids:
        pagc_results[qid] = linear_aggregation([rg_yn_results[qid], gccp_results[qid]])

    test_qrels = {qid: qrels[qid] for qid in all_qids if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(
        test_qrels, {"ndcg_cut_10", "P_10", "recall_10"}
    )

    e5_run = {
        qid: {doc["docid"]: doc["score"] for doc in e5_results[qid][:100]}
        for qid in all_qids
    }

    e5_eval = evaluator.evaluate(e5_run)
    rg_yn_eval = evaluator.evaluate(rg_yn_results)
    gccp_eval = evaluator.evaluate(gccp_results)
    pagc_eval = evaluator.evaluate(pagc_results)

    def avg(results, metric):
        return sum(r[metric] for r in results.values()) / len(results)

    elapsed = datetime.now() - start_time

    print("\n" + "=" * 70)
    print(f"Results: {dataset.upper()} - {model_name} ({len(all_qids)} queries)")
    print(f"Time: {elapsed}")
    print("=" * 70)

    print(f"\n{'Method':<20} {'NDCG@10':>12} {'P@10':>12} {'Recall@10':>12}")
    print("-" * 60)
    print(f"{'E5':<20} {avg(e5_eval, 'ndcg_cut_10'):>12.4f} {avg(e5_eval, 'P_10'):>12.4f} {avg(e5_eval, 'recall_10'):>12.4f}")
    print(f"{'RG-YN':<20} {avg(rg_yn_eval, 'ndcg_cut_10'):>12.4f} {avg(rg_yn_eval, 'P_10'):>12.4f} {avg(rg_yn_eval, 'recall_10'):>12.4f}")
    print(f"{'GCCP':<20} {avg(gccp_eval, 'ndcg_cut_10'):>12.4f} {avg(gccp_eval, 'P_10'):>12.4f} {avg(gccp_eval, 'recall_10'):>12.4f}")
    print(f"{'PAGC (RG-YN+GCCP)':<20} {avg(pagc_eval, 'ndcg_cut_10'):>12.4f} {avg(pagc_eval, 'P_10'):>12.4f} {avg(pagc_eval, 'recall_10'):>12.4f}")

    if output_dir is None:
        output_dir = f"results/trec-dl/{dataset}/{model_name}_e5"
    os.makedirs(output_dir, exist_ok=True)

    for name, data in [("rg_yn", rg_yn_results), ("gccp", gccp_results), ("pagc", pagc_results)]:
        with open(f"{output_dir}/{name}_scores.json", "w") as f:
            json.dump(data, f, indent=2)

    metrics = {
        "experiment": {
            "dataset": dataset,
            "model": model_name,
            "retrieval": "e5",
            "num_queries": len(all_qids),
            "elapsed": str(elapsed),
            "timestamp": start_time.isoformat(),
        },
        "results": {
            "e5": {"ndcg@10": avg(e5_eval, "ndcg_cut_10"), "p@10": avg(e5_eval, "P_10")},
            "rg_yn": {"ndcg@10": avg(rg_yn_eval, "ndcg_cut_10"), "p@10": avg(rg_yn_eval, "P_10")},
            "gccp": {"ndcg@10": avg(gccp_eval, "ndcg_cut_10"), "p@10": avg(gccp_eval, "P_10")},
            "pagc": {"ndcg@10": avg(pagc_eval, "ndcg_cut_10"), "p@10": avg(pagc_eval, "P_10")},
        },
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run GCCP experiment with E5 candidates")
    parser.add_argument("--dataset", type=str, required=True, choices=["dl19", "dl20"])
    parser.add_argument(
        "--model",
        type=str,
        default="flan-t5-large",
        choices=["flan-t5-large", "flan-t5-xl", "flan-ul2"],
    )
    parser.add_argument("--retrieval_file", type=str, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        model_name=args.model,
        retrieval_file=args.retrieval_file,
        num_queries=args.num_queries,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
