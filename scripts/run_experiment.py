#!/usr/bin/env python3
"""
run_experiment.py - Main experiment runner for GCCP reproducibility study

Usage:
    python scripts/run_experiment.py --dataset dl19 --model flan-t5-large
    python scripts/run_experiment.py --dataset dl20 --model flan-t5-xl
    python scripts/run_experiment.py --dataset dl19 --model flan-t5-large --num_queries 5

Arguments:
    --dataset: dl19 or dl20
    --model: flan-t5-large, flan-t5-xl, or flan-ul2
    --num_queries: Optional, limit to N queries for quick testing
    --output_dir: Optional, custom output directory
"""
import sys
import json
import os
import argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
import pytrec_eval


MODEL_MAP = {
    'flan-t5-large': 'google/flan-t5-large',
    'flan-t5-xl': 'google/flan-t5-xl',
    'flan-ul2': 'google/flan-ul2',
}


def load_data(dataset: str, use_pyserini: bool = False):
    """Load queries, qrels, and BM25 results for a dataset.
    
    Args:
        dataset: 'dl19' or 'dl20'
        use_pyserini: If True, use pyserini BM25 with k1=0.9, b=0.4 (paper settings)
    """
    with open(f'data/{dataset}_queries.json', 'r') as f:
        queries = json.load(f)
    with open(f'data/{dataset}_qrels.json', 'r') as f:
        qrels = json.load(f)
    
    if use_pyserini:
        # Pyserini BM25 with paper's exact settings: k1=0.9, b=0.4
        with open(f'data/{dataset}_pyserini_bm25.json', 'r') as f:
            raw_results = json.load(f)
        # Convert to expected format: {qid: [{'docid': ..., 'score': ..., 'contents': ...}]}
        bm25_results = {}
        for qid, data in raw_results.items():
            bm25_results[qid] = [
                {'docid': p['pid'], 'score': 100 - i, 'contents': p['text']}
                for i, p in enumerate(data['passages'])
            ]
    else:
        # Original rank_bm25 results
        with open(f'data/{dataset}/bm25_results.json', 'r') as f:
            bm25_results = json.load(f)
    
    return queries, qrels, bm25_results


def run_experiment(dataset: str, model_name: str, num_queries: int = None, 
                   output_dir: str = None, use_pyserini: bool = False):
    """
    Run GCCP experiment on specified dataset with given model.
    
    Args:
        dataset: 'dl19' or 'dl20'
        model_name: Short model name (e.g., 'flan-t5-large')
        num_queries: Optional limit for quick testing
        output_dir: Optional custom output directory
        use_pyserini: Use pyserini BM25 with paper settings (k1=0.9, b=0.4)
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print(f"GCCP Experiment: {dataset.upper()} with {model_name}")
    bm25_type = "pyserini (k1=0.9, b=0.4)" if use_pyserini else "rank_bm25"
    print(f"BM25 Retrieval: {bm25_type}")
    print("=" * 70)
    print(f"Start time: {start_time}")
    
    # Load data
    queries, qrels, bm25_results = load_data(dataset, use_pyserini)
    
    # Limit queries if specified
    all_qids = list(queries.keys())
    if num_queries:
        all_qids = all_qids[:num_queries]
    
    print(f"Dataset: {dataset.upper()}")
    print(f"Queries: {len(all_qids)}" + (f" (limited from {len(queries)})" if num_queries else ""))
    
    # Initialize models
    full_model_name = MODEL_MAP.get(model_name, model_name)
    print(f"\nLoading model: {full_model_name}")
    
    from src.pointwise.rankers import RGYNRanker
    from src.gccp.gccp_ranker import GCCPRanker
    from src.pagc.aggregation import linear_aggregation
    
    # Author's hyperparameters
    rg_yn_ranker = RGYNRanker(
        full_model_name, 
        template_idx=0,
        target_tokens=('yes', 'no'),
        max_doc_length=128
    )
    
    gccp_ranker = GCCPRanker(
        full_model_name,
        max_doc_length=128,
        m=10,  # Top-10 docs for anchor
        z=10,  # 10 sentences in anchor
        threshold=0.2,
        use_spacy=False
    )
    
    # Run scoring
    rg_yn_results = {}
    gccp_results = {}
    
    for qid in tqdm(all_qids, desc="Processing queries"):
        query = queries[qid]
        docs = bm25_results[qid][:100]
        
        # RG-YN scoring
        rg_yn_rankings = rg_yn_ranker.rank(query, docs)
        rg_yn_results[qid] = {docid: score for docid, score in rg_yn_rankings}
        
        # GCCP scoring
        gccp_rankings, _ = gccp_ranker.rank(query, docs)
        gccp_results[qid] = {docid: score for docid, score in gccp_rankings}
    
    # PAGC aggregation
    pagc_results = {}
    for qid in all_qids:
        pagc_results[qid] = linear_aggregation([rg_yn_results[qid], gccp_results[qid]])
    
    # Evaluate
    test_qrels = {qid: qrels[qid] for qid in all_qids if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(
        test_qrels, {'ndcg_cut_10', 'P_10', 'recall_10'}
    )
    
    bm25_run = {qid: {doc['docid']: doc['score'] for doc in bm25_results[qid][:100]} 
                for qid in all_qids}
    
    bm25_eval = evaluator.evaluate(bm25_run)
    rg_yn_eval = evaluator.evaluate(rg_yn_results)
    gccp_eval = evaluator.evaluate(gccp_results)
    pagc_eval = evaluator.evaluate(pagc_results)
    
    def avg(results, metric):
        return sum(r[metric] for r in results.values()) / len(results)
    
    # Print results
    elapsed = datetime.now() - start_time
    
    print("\n" + "=" * 70)
    print(f"Results: {dataset.upper()} - {model_name} ({len(all_qids)} queries)")
    print(f"Time: {elapsed}")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'NDCG@10':>12} {'P@10':>12} {'Recall@10':>12}")
    print("-" * 60)
    print(f"{'BM25':<20} {avg(bm25_eval, 'ndcg_cut_10'):>12.4f} {avg(bm25_eval, 'P_10'):>12.4f} {avg(bm25_eval, 'recall_10'):>12.4f}")
    print(f"{'RG-YN':<20} {avg(rg_yn_eval, 'ndcg_cut_10'):>12.4f} {avg(rg_yn_eval, 'P_10'):>12.4f} {avg(rg_yn_eval, 'recall_10'):>12.4f}")
    print(f"{'GCCP':<20} {avg(gccp_eval, 'ndcg_cut_10'):>12.4f} {avg(gccp_eval, 'P_10'):>12.4f} {avg(gccp_eval, 'recall_10'):>12.4f}")
    print(f"{'PAGC (RG-YN+GCCP)':<20} {avg(pagc_eval, 'ndcg_cut_10'):>12.4f} {avg(pagc_eval, 'P_10'):>12.4f} {avg(pagc_eval, 'recall_10'):>12.4f}")
    
    # Save results
    if output_dir is None:
        output_dir = f'results/{dataset}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in [('rg_yn', rg_yn_results), ('gccp', gccp_results), ('pagc', pagc_results)]:
        with open(f'{output_dir}/{name}_scores.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    metrics = {
        'experiment': {
            'dataset': dataset,
            'model': model_name,
            'num_queries': len(all_qids),
            'elapsed': str(elapsed),
            'timestamp': start_time.isoformat()
        },
        'results': {
            'bm25': {'ndcg@10': avg(bm25_eval, 'ndcg_cut_10'), 'p@10': avg(bm25_eval, 'P_10')},
            'rg_yn': {'ndcg@10': avg(rg_yn_eval, 'ndcg_cut_10'), 'p@10': avg(rg_yn_eval, 'P_10')},
            'gccp': {'ndcg@10': avg(gccp_eval, 'ndcg_cut_10'), 'p@10': avg(gccp_eval, 'P_10')},
            'pagc': {'ndcg@10': avg(pagc_eval, 'ndcg_cut_10'), 'p@10': avg(pagc_eval, 'P_10')},
        }
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run GCCP experiment')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['dl19', 'dl20'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='flan-t5-large',
                        choices=['flan-t5-large', 'flan-t5-xl', 'flan-ul2'],
                        help='Model to use')
    parser.add_argument('--num_queries', type=int, default=None,
                        help='Limit number of queries (for testing)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Custom output directory')
    parser.add_argument('--use_pyserini', action='store_true',
                        help='Use pyserini BM25 with paper settings (k1=0.9, b=0.4)')
    
    args = parser.parse_args()
    run_experiment(args.dataset, args.model, args.num_queries, args.output_dir, args.use_pyserini)


if __name__ == "__main__":
    main()
