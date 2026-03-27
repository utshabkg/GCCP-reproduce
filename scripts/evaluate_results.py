#!/usr/bin/env python3
"""
evaluate_results.py - Evaluate saved experiment results

Usage:
    python scripts/evaluate_results.py --results_dir results/dl19/flan-t5-large
    python scripts/evaluate_results.py --results_dir results/dl20/flan-t5-xl --compare_paper
"""
import sys
import json
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytrec_eval


# Paper reported results (SIGIR 2025)
PAPER_RESULTS = {
    'dl19': {
        'flan-t5-large': {'bm25': 0.5058, 'rg_yn': 0.6643, 'gccp': 0.6480, 'pagc': 0.7012},
        'flan-t5-xl': {'bm25': 0.5058, 'rg_yn': 0.6731, 'gccp': 0.6579, 'pagc': 0.7103},
    },
    'dl20': {
        'flan-t5-large': {'bm25': 0.4796, 'rg_yn': 0.6493, 'gccp': 0.6570, 'pagc': 0.6910},
        'flan-t5-xl': {'bm25': 0.4796, 'rg_yn': 0.6730, 'gccp': 0.6670, 'pagc': 0.7071},
    }
}


def evaluate_results(results_dir: str, compare_paper: bool = False):
    """Evaluate and display results from a saved experiment."""
    
    # Load metrics
    metrics_path = os.path.join(results_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("=" * 70)
        print(f"Experiment: {metrics['experiment']['dataset'].upper()} - {metrics['experiment']['model']}")
        print(f"Queries: {metrics['experiment']['num_queries']}")
        print(f"Time: {metrics['experiment']['elapsed']}")
        print("=" * 70)
        
        print(f"\n{'Method':<20} {'NDCG@10':>12} {'P@10':>12}")
        print("-" * 50)
        for method in ['bm25', 'rg_yn', 'gccp', 'pagc']:
            r = metrics['results'][method]
            print(f"{method.upper():<20} {r['ndcg@10']:>12.4f} {r['p@10']:>12.4f}")
        
        if compare_paper:
            dataset = metrics['experiment']['dataset']
            model = metrics['experiment']['model']
            
            if dataset in PAPER_RESULTS and model in PAPER_RESULTS[dataset]:
                paper = PAPER_RESULTS[dataset][model]
                print("\n" + "-" * 50)
                print("Comparison with Paper:")
                print(f"{'Method':<20} {'Ours':>12} {'Paper':>12} {'Diff':>12}")
                print("-" * 50)
                for method in ['bm25', 'rg_yn', 'gccp', 'pagc']:
                    ours = metrics['results'][method]['ndcg@10']
                    theirs = paper[method]
                    diff = ours - theirs
                    print(f"{method.upper():<20} {ours:>12.4f} {theirs:>12.4f} {diff:>+12.4f}")
    else:
        print(f"No metrics.json found in {results_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--compare_paper', action='store_true',
                        help='Compare with paper-reported results')
    
    args = parser.parse_args()
    evaluate_results(args.results_dir, args.compare_paper)


if __name__ == "__main__":
    main()
