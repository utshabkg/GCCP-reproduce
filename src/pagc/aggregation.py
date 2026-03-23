"""
PAGC: Post-Aggregation with Global Context

Implements score aggregation from multiple pointwise methods (Eq. 11).
"""
from typing import Dict, List, Tuple, Optional
import numpy as np


def linear_aggregation(scores_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Linear score aggregation (Eq. 11).
    
    f_final(q, d_i) = (1 / |R|+1) * (Σ_R f(q, d_i) + f_c(q, d_i, d_a))
    
    Args:
        scores_list: List of score dicts, each mapping docid -> score
                    Last element should be GCCP scores
                    
    Returns:
        Dict mapping docid -> aggregated score
    """
    if not scores_list:
        return {}
    
    # Get all document IDs
    all_docids = set()
    for scores in scores_list:
        all_docids.update(scores.keys())
    
    # Normalize each score dict to [0, 1] range
    normalized_scores = []
    for scores in scores_list:
        if len(scores) == 0:
            normalized_scores.append({})
            continue
            
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val - min_val > 0:
            norm = {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
        else:
            norm = {k: 0.5 for k in scores.keys()}
        
        normalized_scores.append(norm)
    
    # Average scores (Eq. 11)
    n_methods = len(scores_list)
    aggregated = {}
    
    for docid in all_docids:
        total = 0
        count = 0
        for scores in normalized_scores:
            if docid in scores:
                total += scores[docid]
                count += 1
        
        aggregated[docid] = total / n_methods if n_methods > 0 else 0
    
    return aggregated


def borda_aggregation(rankings_list: List[List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    Borda count aggregation.
    
    Args:
        rankings_list: List of rankings, each a list of (docid, score) sorted by score desc
        
    Returns:
        Dict mapping docid -> Borda score
    """
    borda_scores = {}
    
    for ranking in rankings_list:
        n = len(ranking)
        for rank, (docid, _) in enumerate(ranking):
            if docid not in borda_scores:
                borda_scores[docid] = 0
            # Higher rank (lower index) gets higher score
            borda_scores[docid] += n - rank
    
    return borda_scores


def condorcet_aggregation(rankings_list: List[List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    Condorcet-based aggregation (pairwise wins).
    
    Args:
        rankings_list: List of rankings
        
    Returns:
        Dict mapping docid -> Condorcet score
    """
    # Get all docids
    all_docids = set()
    for ranking in rankings_list:
        all_docids.update([docid for docid, _ in ranking])
    
    all_docids = list(all_docids)
    n = len(all_docids)
    
    # Build pairwise comparison matrix
    wins = {docid: 0 for docid in all_docids}
    
    for ranking in rankings_list:
        # Create rank mapping
        rank_map = {docid: rank for rank, (docid, _) in enumerate(ranking)}
        
        # Compare all pairs
        for i, doc_i in enumerate(all_docids):
            for j, doc_j in enumerate(all_docids):
                if i >= j:
                    continue
                
                rank_i = rank_map.get(doc_i, float('inf'))
                rank_j = rank_map.get(doc_j, float('inf'))
                
                if rank_i < rank_j:
                    wins[doc_i] += 1
                elif rank_j < rank_i:
                    wins[doc_j] += 1
    
    return wins


def copeland_aggregation(rankings_list: List[List[Tuple[str, float]]]) -> Dict[str, float]:
    """
    Copeland aggregation (wins - losses).
    
    Args:
        rankings_list: List of rankings
        
    Returns:
        Dict mapping docid -> Copeland score
    """
    all_docids = set()
    for ranking in rankings_list:
        all_docids.update([docid for docid, _ in ranking])
    
    all_docids = list(all_docids)
    
    # Build win/loss matrix
    scores = {docid: 0 for docid in all_docids}
    
    for ranking in rankings_list:
        rank_map = {docid: rank for rank, (docid, _) in enumerate(ranking)}
        
        for i, doc_i in enumerate(all_docids):
            for j, doc_j in enumerate(all_docids):
                if i >= j:
                    continue
                
                rank_i = rank_map.get(doc_i, float('inf'))
                rank_j = rank_map.get(doc_j, float('inf'))
                
                if rank_i < rank_j:
                    scores[doc_i] += 1
                    scores[doc_j] -= 1
                elif rank_j < rank_i:
                    scores[doc_j] += 1
                    scores[doc_i] -= 1
    
    return scores


class PAGCAggregator:
    """
    Post-Aggregation with Global Context.
    
    Combines scores from multiple pointwise methods with GCCP.
    """
    
    def __init__(self, method: str = 'linear'):
        """
        Initialize aggregator.
        
        Args:
            method: Aggregation method ('linear', 'borda', 'condorcet', 'copeland')
        """
        self.method = method
    
    def aggregate(self, pointwise_scores: Dict[str, Dict[str, float]],
                  gccp_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate pointwise scores with GCCP scores.
        
        Args:
            pointwise_scores: Dict mapping method_name -> (docid -> score)
            gccp_scores: Dict mapping docid -> GCCP score
            
        Returns:
            Dict mapping docid -> aggregated score
        """
        # Combine all scores
        all_scores = list(pointwise_scores.values()) + [gccp_scores]
        
        if self.method == 'linear':
            return linear_aggregation(all_scores)
        
        elif self.method in ['borda', 'condorcet', 'copeland']:
            # Convert scores to rankings
            rankings = []
            for scores in all_scores:
                ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                rankings.append(ranking)
            
            if self.method == 'borda':
                return borda_aggregation(rankings)
            elif self.method == 'condorcet':
                return condorcet_aggregation(rankings)
            else:
                return copeland_aggregation(rankings)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def aggregate_and_rank(self, pointwise_scores: Dict[str, Dict[str, float]],
                           gccp_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Aggregate and return sorted ranking.
        
        Returns:
            List of (docid, score) sorted by score descending
        """
        aggregated = self.aggregate(pointwise_scores, gccp_scores)
        return sorted(aggregated.items(), key=lambda x: x[1], reverse=True)


def pagc_qyg(qg_scores: Dict[str, float], 
             rg_yn_scores: Dict[str, float],
             gccp_scores: Dict[str, float],
             method: str = 'linear') -> List[Tuple[str, float]]:
    """
    PAGC-QYG: Aggregation of QG + RG-YN + GCCP.
    
    Args:
        qg_scores: Query Generation scores
        rg_yn_scores: RG-YN scores
        gccp_scores: GCCP scores
        method: Aggregation method
        
    Returns:
        Sorted ranking
    """
    aggregator = PAGCAggregator(method=method)
    return aggregator.aggregate_and_rank(
        {'qg': qg_scores, 'rg_yn': rg_yn_scores},
        gccp_scores
    )


def pagc_qsg(qg_scores: Dict[str, float],
             rg_s_scores: Dict[str, float], 
             gccp_scores: Dict[str, float],
             method: str = 'linear') -> List[Tuple[str, float]]:
    """
    PAGC-QSG: Aggregation of QG + RG-S + GCCP.
    """
    aggregator = PAGCAggregator(method=method)
    return aggregator.aggregate_and_rank(
        {'qg': qg_scores, 'rg_s': rg_s_scores},
        gccp_scores
    )


if __name__ == "__main__":
    # Test aggregation
    scores_a = {'doc1': 0.9, 'doc2': 0.7, 'doc3': 0.5}
    scores_b = {'doc1': 0.8, 'doc2': 0.85, 'doc3': 0.6}
    scores_gccp = {'doc1': 0.95, 'doc2': 0.6, 'doc3': 0.7}
    
    print("Testing Linear Aggregation...")
    aggregated = linear_aggregation([scores_a, scores_b, scores_gccp])
    print(f"Aggregated: {aggregated}")
    
    print("\nTesting PAGC Aggregator...")
    aggregator = PAGCAggregator(method='linear')
    ranking = aggregator.aggregate_and_rank(
        {'method_a': scores_a, 'method_b': scores_b},
        scores_gccp
    )
    print(f"Ranking: {ranking}")
