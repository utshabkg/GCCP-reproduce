#!/usr/bin/env python
"""
Main Experiment Runner for GCCP Reproducibility Study

Usage:
    python run_experiment.py --dataset dl19 --model google/flan-t5-xl --method gccp
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.bm25_retrieval import BM25Retriever, load_trec_queries, load_trec_qrels
from src.pointwise.rankers import RGYNRanker, RGSRanker, QGRanker
from src.gccp.gccp_ranker import GCCPRanker
from src.pagc.aggregation import PAGCAggregator
from src.evaluation.metrics import evaluate_rankings, format_results


def setup_args():
    parser = argparse.ArgumentParser(description='GCCP Reproducibility Experiments')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='dl19',
                       choices=['dl19', 'dl20'],
                       help='Dataset to evaluate on')
    
    # Model
    parser.add_argument('--model', type=str, default='google/flan-t5-large',
                       help='Model name or path')
    
    # Method
    parser.add_argument('--method', type=str, default='all',
                       choices=['bm25', 'rg_yn', 'rg_s', 'qg', 'gccp', 'pagc_qyg', 'pagc_qsg', 'all'],
                       help='Ranking method to use')
    
    # GCCP parameters
    parser.add_argument('--m', type=int, default=10,
                       help='Number of top docs for anchor generation')
    parser.add_argument('--z', type=int, default=10,
                       help='Number of sentences in anchor')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Similarity threshold for MDS')
    
    # Retrieval
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of documents to retrieve/rerank')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    # Other
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--use_spacy', action='store_true', default=True,
                       help='Use spaCy for sentence segmentation')
    
    return parser.parse_args()


def run_bm25(dataset: str, top_k: int = 100) -> Tuple[Dict, Dict, Dict]:
    """Run BM25 retrieval and return queries, qrels, and retrieved docs."""
    print(f"\n{'='*50}")
    print(f"Running BM25 retrieval for {dataset}")
    print(f"{'='*50}")
    
    # Load queries and qrels
    queries = load_trec_queries(dataset)
    qrels = load_trec_qrels(dataset)
    
    print(f"Loaded {len(queries)} queries")
    
    # Run retrieval
    retriever = BM25Retriever(dataset)
    retrieved = retriever.batch_retrieve(queries, top_k=top_k)
    
    # Convert to rankings for evaluation
    rankings = {}
    for qid, docs in retrieved.items():
        rankings[qid] = [(d['docid'], d['score']) for d in docs]
    
    # Evaluate BM25
    results = evaluate_rankings(rankings, qrels)
    print(f"\nBM25 Results:")
    print(format_results(results))
    
    return queries, qrels, retrieved


def run_pointwise(method: str, model_name: str, 
                  queries: Dict, retrieved: Dict,
                  qrels: Dict, device: str = None) -> Dict:
    """Run pointwise ranking method."""
    print(f"\n{'='*50}")
    print(f"Running {method.upper()} with {model_name}")
    print(f"{'='*50}")
    
    # Initialize ranker
    if method == 'rg_yn':
        ranker = RGYNRanker(model_name, device=device)
    elif method == 'rg_s':
        ranker = RGSRanker(model_name, device=device)
    elif method == 'qg':
        ranker = QGRanker(model_name, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Rank for each query
    rankings = {}
    all_scores = {}
    
    for qid, query in tqdm(queries.items(), desc=f"{method.upper()} Ranking"):
        docs = retrieved[qid]
        ranking = ranker.rank(query, docs)
        rankings[qid] = ranking
        all_scores[qid] = {docid: score for docid, score in ranking}
    
    # Evaluate
    results = evaluate_rankings(rankings, qrels)
    print(f"\n{method.upper()} Results:")
    print(format_results(results))
    
    return {'rankings': rankings, 'scores': all_scores, 'results': results}


def run_gccp(model_name: str, queries: Dict, retrieved: Dict, qrels: Dict,
             m: int = 10, z: int = 10, threshold: float = 0.1,
             device: str = None, use_spacy: bool = True) -> Dict:
    """Run GCCP ranking."""
    print(f"\n{'='*50}")
    print(f"Running GCCP with {model_name}")
    print(f"{'='*50}")
    
    ranker = GCCPRanker(
        model_name, 
        device=device, 
        m=m, z=z, 
        threshold=threshold,
        use_spacy=use_spacy
    )
    
    rankings = {}
    all_scores = {}
    anchors = {}
    
    for qid, query in tqdm(queries.items(), desc="GCCP Ranking"):
        docs = retrieved[qid]
        ranking, anchor = ranker.rank(query, docs)
        rankings[qid] = ranking
        all_scores[qid] = {docid: score for docid, score in ranking}
        anchors[qid] = anchor
    
    # Evaluate
    results = evaluate_rankings(rankings, qrels)
    print(f"\nGCCP Results:")
    print(format_results(results))
    
    return {'rankings': rankings, 'scores': all_scores, 'anchors': anchors, 'results': results}


def run_pagc(pointwise_results: Dict, gccp_results: Dict, qrels: Dict,
             method: str = 'linear') -> Dict:
    """Run PAGC aggregation."""
    print(f"\n{'='*50}")
    print(f"Running PAGC Aggregation ({method})")
    print(f"{'='*50}")
    
    aggregator = PAGCAggregator(method=method)
    
    rankings = {}
    for qid in gccp_results['scores'].keys():
        # Gather scores from all methods
        pointwise_scores = {}
        for pw_name, pw_result in pointwise_results.items():
            if qid in pw_result['scores']:
                pointwise_scores[pw_name] = pw_result['scores'][qid]
        
        gccp_scores = gccp_results['scores'][qid]
        
        # Aggregate
        ranking = aggregator.aggregate_and_rank(pointwise_scores, gccp_scores)
        rankings[qid] = ranking
    
    # Evaluate
    results = evaluate_rankings(rankings, qrels)
    print(f"\nPAGC Results:")
    print(format_results(results))
    
    return {'rankings': rankings, 'results': results}


def save_results(results: Dict, output_dir: str, dataset: str, 
                 model_name: str, method: str):
    """Save results to file."""
    output_path = Path(output_dir) / dataset / model_name.replace('/', '_')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save rankings
    rankings_file = output_path / f"{method}_rankings.json"
    with open(rankings_file, 'w') as f:
        # Convert rankings to serializable format
        serializable = {}
        for qid, ranking in results.get('rankings', {}).items():
            serializable[qid] = [[docid, float(score)] for docid, score in ranking]
        json.dump(serializable, f, indent=2)
    
    # Save metrics
    metrics_file = output_path / f"{method}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results.get('results', {}), f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    args = setup_args()
    
    print(f"\n{'#'*60}")
    print(f"GCCP Reproducibility Experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"{'#'*60}")
    
    # Run BM25 retrieval
    queries, qrels, retrieved = run_bm25(args.dataset, args.top_k)
    
    if args.method == 'bm25':
        return
    
    # Store results for aggregation
    pointwise_results = {}
    gccp_results = None
    
    # Run requested methods
    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['rg_yn', 'rg_s', 'qg', 'gccp']
    elif args.method.startswith('pagc'):
        # Need pointwise methods for PAGC
        if 'qyg' in args.method:
            methods_to_run = ['qg', 'rg_yn', 'gccp']
        else:
            methods_to_run = ['qg', 'rg_s', 'gccp']
    else:
        methods_to_run = [args.method]
    
    # Run pointwise methods
    for method in methods_to_run:
        if method in ['rg_yn', 'rg_s', 'qg']:
            result = run_pointwise(
                method, args.model, queries, retrieved, qrels, args.device
            )
            pointwise_results[method] = result
            save_results(result, args.output_dir, args.dataset, args.model, method)
        
        elif method == 'gccp':
            gccp_results = run_gccp(
                args.model, queries, retrieved, qrels,
                m=args.m, z=args.z, threshold=args.threshold,
                device=args.device, use_spacy=args.use_spacy
            )
            save_results(gccp_results, args.output_dir, args.dataset, args.model, 'gccp')
    
    # Run PAGC if requested
    if args.method in ['pagc_qyg', 'pagc_qsg', 'all']:
        if gccp_results is not None and pointwise_results:
            if args.method == 'pagc_qyg' or args.method == 'all':
                qyg_pw = {k: v for k, v in pointwise_results.items() if k in ['qg', 'rg_yn']}
                if qyg_pw:
                    pagc_result = run_pagc(qyg_pw, gccp_results, qrels)
                    save_results(pagc_result, args.output_dir, args.dataset, args.model, 'pagc_qyg')
            
            if args.method == 'pagc_qsg' or args.method == 'all':
                qsg_pw = {k: v for k, v in pointwise_results.items() if k in ['qg', 'rg_s']}
                if qsg_pw:
                    pagc_result = run_pagc(qsg_pw, gccp_results, qrels)
                    save_results(pagc_result, args.output_dir, args.dataset, args.model, 'pagc_qsg')
    
    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
