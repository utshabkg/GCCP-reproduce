"""
BM25 First-Stage Retrieval

Option 1: Use Pyserini (if Java 21+ available)
Option 2: Use rank_bm25 (pure Python fallback)
Option 3: Load pre-computed BM25 results
"""
import os
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import ir_datasets

# Try to import pyserini, fall back to rank_bm25 if not available
PYSERINI_AVAILABLE = False
try:
    from pyserini.search.lucene import LuceneSearcher
    PYSERINI_AVAILABLE = True
except Exception as e:
    print(f"Pyserini not available: {e}")
    print("Will use pre-computed BM25 results or rank_bm25 fallback")


class BM25Retriever:
    """BM25 retriever with multiple backends."""
    
    INDEX_MAP = {
        'msmarco-passage': 'msmarco-v1-passage',
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
    }
    
    def __init__(self, dataset_name: str, precomputed_path: str = None):
        """
        Initialize BM25 retriever.
        
        Args:
            dataset_name: Name of the dataset
            precomputed_path: Path to pre-computed BM25 results JSON
        """
        self.dataset_name = dataset_name
        self.precomputed_path = precomputed_path
        self.precomputed_results = None
        self.searcher = None
        self.corpus = None
        
        # Try to load pre-computed results first
        if precomputed_path and os.path.exists(precomputed_path):
            print(f"Loading pre-computed BM25 results from {precomputed_path}")
            with open(precomputed_path, 'r') as f:
                self.precomputed_results = json.load(f)
        elif PYSERINI_AVAILABLE:
            index_name = self.INDEX_MAP.get(dataset_name.lower())
            if index_name:
                print(f"Using Pyserini with index: {index_name}")
                self.searcher = LuceneSearcher.from_prebuilt_index(index_name)
                self.searcher.set_bm25(k1=0.9, b=0.4)
        else:
            print("Loading corpus for rank_bm25 fallback...")
            self._load_corpus()
    
    def _load_corpus(self):
        """Load MS MARCO corpus for rank_bm25."""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            nltk.download('punkt', quiet=True)
            
            ds = ir_datasets.load('msmarco-passage')
            self.corpus = {}
            for doc in tqdm(ds.docs_iter(), desc="Loading corpus"):
                self.corpus[doc.doc_id] = doc.text
            
            # Build BM25 index
            doc_ids = list(self.corpus.keys())
            tokenized = [self.corpus[did].lower().split() for did in doc_ids]
            self.bm25 = BM25Okapi(tokenized)
            self.doc_ids = doc_ids
        except ImportError:
            print("rank_bm25 not installed. Install with: pip install rank_bm25")
            self.corpus = None
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        """Retrieve top-k documents for a query."""
        if self.precomputed_results:
            # This won't work with just a query string
            raise ValueError("Use batch_retrieve with precomputed results")
        
        if self.searcher:
            hits = self.searcher.search(query, k=top_k)
            results = []
            for hit in hits:
                doc = self.searcher.doc(hit.docid)
                results.append({
                    'docid': hit.docid,
                    'score': hit.score,
                    'contents': doc.raw() if doc else ""
                })
            return results
        
        if self.corpus and hasattr(self, 'bm25'):
            # Use rank_bm25 fallback
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_idx = scores.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_idx:
                doc_id = self.doc_ids[idx]
                results.append({
                    'docid': doc_id,
                    'score': float(scores[idx]),
                    'contents': self.corpus[doc_id]
                })
            return results
        
        raise RuntimeError("No retrieval backend available")
    
    def batch_retrieve(self, queries: Dict[str, str], top_k: int = 100,
                       output_path: str = None) -> Dict[str, List[Dict]]:
        """Retrieve documents for multiple queries."""
        if self.precomputed_results:
            # Return pre-computed results
            return {qid: self.precomputed_results.get(qid, [])[:top_k] 
                   for qid in queries.keys()}
        
        results = {}
        for qid, query in tqdm(queries.items(), desc="BM25 Retrieval"):
            results[qid] = self.retrieve(query, top_k)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results


def load_trec_queries(dataset: str) -> Dict[str, str]:
    """Load TREC DL queries using ir_datasets."""
    dataset_map = {
        'dl19': 'msmarco-passage/trec-dl-2019/judged',
        'dl20': 'msmarco-passage/trec-dl-2020/judged',
    }
    
    ds = ir_datasets.load(dataset_map[dataset])
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    return queries


def load_trec_qrels(dataset: str) -> Dict[str, Dict[str, int]]:
    """Load TREC DL relevance judgments using ir_datasets."""
    dataset_map = {
        'dl19': 'msmarco-passage/trec-dl-2019/judged',
        'dl20': 'msmarco-passage/trec-dl-2020/judged',
    }
    
    ds = ir_datasets.load(dataset_map[dataset])
    qrels = {}
    for qrel in ds.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    
    return qrels


def load_msmarco_doc(doc_id: str) -> str:
    """Load a single document from MS MARCO."""
    ds = ir_datasets.load('msmarco-passage')
    doc_store = ds.docs_store()
    doc = doc_store.get(doc_id)
    return doc.text if doc else ""


if __name__ == "__main__":
    print("Testing ir_datasets...")
    queries = load_trec_queries('dl19')
    qrels = load_trec_qrels('dl19')
    
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded qrels for {len(qrels)} queries")
    
    sample_qid = list(queries.keys())[0]
    print(f"\nSample query [{sample_qid}]: {queries[sample_qid]}")
