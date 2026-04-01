#!/usr/bin/env python3
"""
BEIR Benchmark Evaluation for GCCP Reproducibility Study

Evaluates on 8 BEIR datasets:
- trec-covid, robust04, webis-touche2020, scifact
- signal1m, trec-news, dbpedia-entity, nfcorpus

Usage:
    python scripts/run_beir.py --dataset scifact --model flan-t5-large
    python scripts/run_beir.py --dataset all --model flan-t5-xl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels

from src.pointwise.rankers import RGYNRanker
from src.gccp.gccp_ranker import GCCPRanker

# BEIR datasets used in the paper
BEIR_DATASETS = [
    'scifact',
    'nfcorpus', 
    'trec-covid',
    'webis-touche2020',
    'dbpedia-entity',
    'robust04',
    'signal1m',
    'trec-news',
]

# Paper's reported results (Table 1 - Flan-T5-Large)
# Values verified from paper Table 1
PAPER_RESULTS = {
    'trec-covid': {'RG-YN': 0.6925, 'GCCP': 0.7580, 'PAGC': 0.7559},
    'robust04': {'RG-YN': 0.4407, 'GCCP': 0.4457, 'PAGC': 0.4752},
    'webis-touche2020': {'RG-YN': 0.2780, 'GCCP': 0.2697, 'PAGC': 0.2614},
    'scifact': {'RG-YN': 0.5379, 'GCCP': 0.5966, 'PAGC': 0.6485},
    'signal1m': {'RG-YN': 0.2914, 'GCCP': 0.3010, 'PAGC': 0.2966},
    'trec-news': {'RG-YN': 0.3534, 'GCCP': 0.4005, 'PAGC': 0.3933},
    'dbpedia-entity': {'RG-YN': 0.3246, 'GCCP': 0.3974, 'PAGC': 0.4054},
    'nfcorpus': {'RG-YN': 0.3282, 'GCCP': 0.3505, 'PAGC': 0.3526},
}


def get_beir_data(dataset_name: str, top_k: int = 100):
    """Load BEIR dataset using pyserini."""
    print(f"\nLoading BEIR dataset: {dataset_name}")
    
    # Index name
    index_name = f'beir-v1.0.0-{dataset_name}.flat'
    topics_name = f'beir-v1.0.0-{dataset_name}-test'
    
    # Load searcher
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    
    # Load topics (queries)
    topics = get_topics(topics_name)
    
    # Load qrels
    qrels = get_qrels(topics_name)
    
    # Get BM25 results for each query
    print(f"Running BM25 retrieval for {len(topics)} queries...")
    queries = {}
    bm25_results = {}
    
    for qid, topic in tqdm(topics.items(), desc="BM25 Search"):
        query = topic['title'] if isinstance(topic, dict) else topic
        queries[qid] = query
        
        # Search
        hits = searcher.search(query, k=top_k)
        
        # Get documents
        docs = []
        for hit in hits:
            doc = searcher.doc(hit.docid)
            if doc:
                raw = json.loads(doc.raw())
                docs.append({
                    'docid': hit.docid,
                    'contents': raw.get('contents', raw.get('text', '')),
                    'score': hit.score
                })
        
        bm25_results[qid] = docs
    
    return queries, bm25_results, qrels


def compute_ndcg(rankings: dict, qrels: dict, k: int = 10) -> float:
    """Compute NDCG@k using pytrec_eval for accuracy."""
    import pytrec_eval
    import numpy as np
    
    # Convert rankings to pytrec_eval format: {qid: {docid: score}}
    run = {}
    for qid, ranking in rankings.items():
        qid_str = str(qid)
        run[qid_str] = {str(docid): float(score) for docid, score in ranking[:100]}
    
    # Convert qrels to pytrec_eval format: {qid: {docid: int_relevance}}
    qrels_dict = {}
    for qid, rels in qrels.items():
        qid_str = str(qid)
        qrels_dict[qid_str] = {str(docid): int(rel) for docid, rel in rels.items()}
    
    # Evaluate using pytrec_eval (matches official trec_eval)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {f'ndcg_cut_{k}'})
    results = evaluator.evaluate(run)
    
    # Get mean NDCG@k
    metric_name = f'ndcg_cut_{k}'
    ndcg_scores = [v[metric_name] for qid, v in results.items() if qid in run]
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def run_beir_experiment(dataset_name: str, model_name: str, output_dir: str):
    """Run GCCP experiment on a BEIR dataset."""
    
    print("=" * 70)
    print(f"BEIR Evaluation: {dataset_name}")
    print(f"Model: {model_name}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Load data
    queries, bm25_results, qrels = get_beir_data(dataset_name)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Queries: {len(queries)}")
    
    # Model mapping
    model_map = {
        'flan-t5-large': 'google/flan-t5-large',
        'flan-t5-xl': 'google/flan-t5-xl',
        'flan-ul2': 'google/flan-ul2',
    }
    hf_model = model_map.get(model_name, model_name)
    
    # Initialize rankers
    print(f"\nLoading model: {hf_model}")
    rgyn_ranker = RGYNRanker(hf_model)
    gccp_ranker = GCCPRanker(hf_model)
    
    # Results storage
    rgyn_rankings = {}
    gccp_rankings = {}
    pagc_rankings = {}
    
    # Process queries
    for qid, query in tqdm(queries.items(), desc="Processing queries"):
        documents = bm25_results[qid]
        
        if not documents or len(documents) < 2:
            continue
        
        try:
            # RG-YN
            rgyn_scores = rgyn_ranker.rank(query, documents)
            rgyn_rankings[qid] = rgyn_scores
            
            # GCCP
            gccp_scores, anchor = gccp_ranker.rank(query, documents)
            gccp_rankings[qid] = gccp_scores
            
            # PAGC (linear combination)
            rgyn_dict = {docid: score for docid, score in rgyn_scores}
            gccp_dict = {docid: score for docid, score in gccp_scores}
        
            pagc_scores = []
            for docid in rgyn_dict:
                combined = rgyn_dict[docid] + gccp_dict.get(docid, 0)
                pagc_scores.append((docid, combined))
            pagc_scores.sort(key=lambda x: x[1], reverse=True)
            pagc_rankings[qid] = pagc_scores
        except Exception as e:
            print(f"Warning: Error processing query {qid}: {e}")
            continue
    
    # Compute BM25 baseline
    bm25_rankings = {
        qid: [(doc['docid'], doc['score']) for doc in docs]
        for qid, docs in bm25_results.items()
    }
    
    # Evaluate
    bm25_ndcg = compute_ndcg(bm25_rankings, qrels)
    rgyn_ndcg = compute_ndcg(rgyn_rankings, qrels)
    gccp_ndcg = compute_ndcg(gccp_rankings, qrels)
    pagc_ndcg = compute_ndcg(pagc_rankings, qrels)
    
    elapsed = datetime.now() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print(f"Results: {dataset_name} - {model_name}")
    print(f"Time: {elapsed}")
    print("=" * 70)
    print(f"\n{'Method':<25} {'NDCG@10':>12}")
    print("-" * 40)
    print(f"{'BM25':<25} {bm25_ndcg:>12.4f}")
    print(f"{'RG-YN':<25} {rgyn_ndcg:>12.4f}")
    print(f"{'GCCP':<25} {gccp_ndcg:>12.4f}")
    print(f"{'PAGC (RG-YN+GCCP)':<25} {pagc_ndcg:>12.4f}")
    
    # Compare with paper
    if dataset_name in PAPER_RESULTS:
        paper = PAPER_RESULTS[dataset_name]
        print(f"\nPaper comparison:")
        print(f"  RG-YN: {rgyn_ndcg:.4f} vs {paper['RG-YN']:.4f} (gap: {(paper['RG-YN']-rgyn_ndcg)*100:.1f}%)")
        print(f"  GCCP:  {gccp_ndcg:.4f} vs {paper['GCCP']:.4f} (gap: {(paper['GCCP']-gccp_ndcg)*100:.1f}%)")
        print(f"  PAGC:  {pagc_ndcg:.4f} vs {paper['PAGC']:.4f} (gap: {(paper['PAGC']-pagc_ndcg)*100:.1f}%)")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'experiment': {
            'dataset': dataset_name,
            'model': model_name,
            'num_queries': len(queries),
            'elapsed': str(elapsed),
            'timestamp': datetime.now().isoformat()
        },
        'results': {
            'bm25': {'ndcg@10': bm25_ndcg},
            'rg_yn': {'ndcg@10': rgyn_ndcg},
            'gccp': {'ndcg@10': gccp_ndcg},
            'pagc': {'ndcg@10': pagc_ndcg}
        }
    }
    
    if dataset_name in PAPER_RESULTS:
        results['paper_comparison'] = PAPER_RESULTS[dataset_name]
    
    # Save with model name in filename
    output_file = os.path.join(output_dir, f'{model_name}_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='BEIR Benchmark Evaluation')
    parser.add_argument('--dataset', type=str, default='scifact',
                        help='BEIR dataset name or "all"')
    parser.add_argument('--model', type=str, default='flan-t5-large',
                        choices=['flan-t5-large', 'flan-t5-xl', 'flan-ul2'])
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = BEIR_DATASETS
    else:
        datasets = [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        # New structure: results/beir/{dataset}/{model}_metrics.json
        output_dir = args.output_dir or f'results/beir/{dataset}'
        
        try:
            results = run_beir_experiment(dataset, args.model, output_dir)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("BEIR Summary")
        print("=" * 70)
        print(f"\n{'Dataset':<20} {'RG-YN':>10} {'GCCP':>10} {'PAGC':>10}")
        print("-" * 55)
        for dataset, res in all_results.items():
            r = res['results']
            print(f"{dataset:<20} {r['rg_yn']['ndcg@10']:>10.4f} {r['gccp']['ndcg@10']:>10.4f} {r['pagc']['ndcg@10']:>10.4f}")


if __name__ == '__main__':
    main()
