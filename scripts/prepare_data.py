#!/usr/bin/env python
"""
Download pre-computed BM25 results from the original GCCP repository.
"""
import os
import json
from pathlib import Path
import ir_datasets
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_bm25_results(dataset: str, output_dir: str = "data", top_k: int = 100):
    """
    Generate BM25 results using rank_bm25 and ir_datasets.
    """
    from rank_bm25 import BM25Okapi
    from src.retrieval.bm25_retrieval import load_trec_queries
    
    output_path = Path(output_dir) / dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "bm25_results.json"
    
    if results_file.exists():
        print(f"BM25 results already exist at {results_file}")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    print(f"Generating BM25 results for {dataset}...")
    
    # Load queries
    queries = load_trec_queries(dataset)
    print(f"Loaded {len(queries)} queries")
    
    # Load MS MARCO passages
    print("Loading MS MARCO passages (this may take a while)...")
    ds = ir_datasets.load('msmarco-passage')
    
    # Build corpus
    corpus = {}
    doc_ids = []
    tokenized_corpus = []
    
    for doc in tqdm(ds.docs_iter(), desc="Loading corpus"):
        corpus[doc.doc_id] = doc.text
        doc_ids.append(doc.doc_id)
        tokenized_corpus.append(doc.text.lower().split())
    
    print(f"Loaded {len(corpus)} passages")
    
    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Retrieve for each query
    results = {}
    for qid, query in tqdm(queries.items(), desc="BM25 Retrieval"):
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results[qid] = []
        for idx in top_indices:
            results[qid].append({
                'docid': doc_ids[idx],
                'score': float(scores[idx]),
                'contents': corpus[doc_ids[idx]]
            })
    
    # Save results
    print(f"Saving results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print("Done!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare BM25 results")
    parser.add_argument('--dataset', type=str, default='dl19',
                       choices=['dl19', 'dl20'])
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--top_k', type=int, default=100)
    
    args = parser.parse_args()
    
    generate_bm25_results(args.dataset, args.output_dir, args.top_k)
