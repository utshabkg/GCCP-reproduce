#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_retrieval import load_trec_queries
from experiments.dense_retrieval.e5_retriever import E5Retriever


def main():
    parser = argparse.ArgumentParser(description="Generate E5 retrieval results for TREC-DL")
    parser.add_argument("--dataset", type=str, required=True, choices=["dl19", "dl20"])
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--index_dir", type=str, default="data/e5_index")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--max_docs", type=int, default=None,
                        help="For smoke tests only")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="Limit number of queries for quick testing")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    queries = load_trec_queries(args.dataset)
    if args.num_queries is not None:
        qids = list(queries.keys())[:args.num_queries]
        queries = {qid: queries[qid] for qid in qids}

    retriever = E5Retriever(
        model_name=args.model,
        index_dir=args.index_dir,
        device=args.device,
    )

    retriever.build_index(rebuild=args.rebuild_index, max_docs=args.max_docs)
    results = retriever.batch_retrieve_pyserini_style(queries, top_k=args.top_k)

    output_file = args.output_file or f"data/{args.dataset}_e5_results.json"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved E5 results to {output_file}")


if __name__ == "__main__":
    main()
