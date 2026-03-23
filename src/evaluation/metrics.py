"""
Evaluation Metrics for IR

Implements NDCG@k, Precision@k, Recall@k using pytrec_eval.
"""
import pytrec_eval
from typing import Dict, List, Tuple
import numpy as np


def convert_ranking_to_run(rankings: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Convert rankings to TREC run format.
    
    Args:
        rankings: Dict mapping qid -> list of (docid, score)
        
    Returns:
        Dict mapping qid -> dict of docid -> score
    """
    run = {}
    for qid, ranking in rankings.items():
        run[qid] = {docid: score for docid, score in ranking}
    return run


def evaluate_rankings(rankings: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]],
                     metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate rankings against qrels.
    
    Args:
        rankings: Dict mapping qid -> list of (docid, score)
        qrels: Dict mapping qid -> dict of docid -> relevance
        metrics: List of metrics to compute
        
    Returns:
        Dict mapping metric -> score
    """
    if metrics is None:
        metrics = ['ndcg_cut_10', 'P_10', 'recall_10']
    
    # Convert rankings to run format
    run = convert_ranking_to_run(rankings)
    
    # Filter qrels to only include queries in run
    filtered_qrels = {qid: rels for qid, rels in qrels.items() if qid in run}
    
    # Create evaluator
    evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, set(metrics))
    
    # Evaluate
    results = evaluator.evaluate(run)
    
    # Aggregate across queries
    aggregated = {}
    for metric in metrics:
        scores = [results[qid][metric] for qid in results]
        aggregated[metric] = np.mean(scores) if scores else 0.0
    
    return aggregated


def compute_ndcg(ranking: List[Tuple[str, float]], 
                 qrels: Dict[str, int], 
                 k: int = 10) -> float:
    """
    Compute NDCG@k for a single query.
    
    Args:
        ranking: List of (docid, score) sorted by score desc
        qrels: Dict mapping docid -> relevance
        k: Cutoff
        
    Returns:
        NDCG@k score
    """
    # Get relevance scores
    rels = []
    for docid, _ in ranking[:k]:
        rel = qrels.get(docid, 0)
        rels.append(rel)
    
    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))
    
    # Ideal DCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_single_query(ranking: List[Tuple[str, float]],
                         qrels: Dict[str, int],
                         k: int = 10) -> Dict[str, float]:
    """
    Evaluate a single query's ranking.
    
    Args:
        ranking: List of (docid, score)
        qrels: Relevance judgments for this query
        k: Cutoff
        
    Returns:
        Dict with NDCG@k, P@k, Recall@k
    """
    # Get top-k docs
    top_k = ranking[:k]
    top_k_docids = set(docid for docid, _ in top_k)
    
    # Relevant docs
    relevant_docs = set(docid for docid, rel in qrels.items() if rel > 0)
    
    # Precision@k
    relevant_in_top_k = len(top_k_docids & relevant_docs)
    precision = relevant_in_top_k / k if k > 0 else 0.0
    
    # Recall@k
    recall = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0.0
    
    # NDCG@k
    ndcg = compute_ndcg(ranking, qrels, k)
    
    return {
        f'ndcg@{k}': ndcg,
        f'precision@{k}': precision,
        f'recall@{k}': recall
    }


def format_results(results: Dict[str, float], 
                   dataset: str = None,
                   method: str = None) -> str:
    """
    Format results for display.
    
    Args:
        results: Dict of metric -> score
        dataset: Dataset name
        method: Method name
        
    Returns:
        Formatted string
    """
    lines = []
    if dataset and method:
        lines.append(f"Results for {method} on {dataset}:")
    
    for metric, score in sorted(results.items()):
        lines.append(f"  {metric}: {score:.4f}")
    
    return "\n".join(lines)


def statistical_significance(scores_a: List[float], 
                            scores_b: List[float],
                            test: str = 'paired_t') -> Tuple[float, bool]:
    """
    Test statistical significance between two sets of scores.
    
    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B
        test: Test type ('paired_t' or 'wilcoxon')
        
    Returns:
        Tuple of (p-value, is_significant at p<0.05)
    """
    from scipy import stats
    
    if test == 'paired_t':
        _, p_value = stats.ttest_rel(scores_a, scores_b)
    elif test == 'wilcoxon':
        _, p_value = stats.wilcoxon(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return p_value, p_value < 0.05


if __name__ == "__main__":
    # Test evaluation
    rankings = {
        'q1': [('d1', 0.9), ('d2', 0.8), ('d3', 0.7)],
        'q2': [('d4', 0.95), ('d5', 0.85), ('d6', 0.75)],
    }
    
    qrels = {
        'q1': {'d1': 3, 'd2': 2, 'd3': 0},
        'q2': {'d4': 3, 'd5': 0, 'd6': 2},
    }
    
    print("Testing evaluation...")
    results = evaluate_rankings(rankings, qrels)
    print(format_results(results, "test", "test_method"))
