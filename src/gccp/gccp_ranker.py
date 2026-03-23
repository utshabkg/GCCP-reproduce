"""
GCCP: Global-Consistent Comparative Pointwise Ranking

Implements contrastive relevance scoring using anchor documents (Eq. 10).
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

from .spectral_mds import generate_anchor_document


# Pairwise comparison prompt (from original paper)
GCCP_PROMPT = """Given a query, which of the following two passages is more relevant to the query?

Query: {query}

Passage A: {passage_a}

Passage B: {passage_b}

The more relevant passage is Passage"""


class GCCPRanker:
    """
    Global-Consistent Comparative Pointwise Ranker.
    
    Uses an anchor document as reference for pairwise comparison
    within a pointwise framework.
    """
    
    def __init__(self, model_name: str, device: str = None,
                 max_length: int = 1024, m: int = 10, z: int = 10,
                 threshold: float = 0.1, use_spacy: bool = True):
        """
        Initialize GCCP ranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            max_length: Maximum input length
            m: Number of top docs for anchor generation
            z: Number of sentences in anchor
            threshold: Similarity threshold for MDS
            use_spacy: Use spaCy for sentence segmentation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.m = m
        self.z = z
        self.threshold = threshold
        self.use_spacy = use_spacy
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if 't5' in model_name.lower() or 'ul2' in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map='auto'
            )
            self.is_encoder_decoder = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map='auto'
            )
            self.is_encoder_decoder = False
        
        self.model.eval()
        
        # Token IDs for "A" and "B"
        self.token_a = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.token_b = self.tokenizer.encode("B", add_special_tokens=False)[0]
    
    def _get_comparison_score(self, query: str, passage: str, anchor: str) -> float:
        """
        Compute contrastive relevance score (Eq. 10).
        
        f_c(q, d_i, d_a) = LLM(d_i | q, d_i, d_a, P_GCCP)
        
        Returns probability that passage is more relevant than anchor.
        """
        # Format prompt with passage as A and anchor as B
        prompt = GCCP_PROMPT.format(
            query=query,
            passage_a=passage[:2000],  # Truncate long passages
            passage_b=anchor[:2000]
        )
        
        inputs = self.tokenizer(
            prompt, return_tensors='pt',
            max_length=self.max_length, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                decoder_input = self.tokenizer(
                    "", return_tensors='pt'
                ).input_ids.to(self.device)
                
                outputs = self.model(
                    **inputs,
                    decoder_input_ids=decoder_input
                )
                logits = outputs.logits[0, -1, :]
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
        
        # Get log probabilities for A and B
        log_probs = F.log_softmax(logits, dim=-1)
        
        score_a = log_probs[self.token_a].item()
        score_b = log_probs[self.token_b].item()
        
        # Return probability of choosing passage (A) over anchor (B)
        prob_a = np.exp(score_a) / (np.exp(score_a) + np.exp(score_b))
        
        return prob_a
    
    def generate_anchor(self, documents: List[Dict]) -> str:
        """
        Generate anchor document from top-m candidates.
        
        Args:
            documents: List of documents (sorted by initial ranking)
            
        Returns:
            Anchor document text
        """
        doc_texts = [d.get('contents', d.get('text', '')) for d in documents[:self.m]]
        
        anchor = generate_anchor_document(
            doc_texts,
            m=self.m,
            z=self.z,
            threshold=self.threshold,
            use_spacy=self.use_spacy
        )
        
        return anchor
    
    def rank(self, query: str, documents: List[Dict], 
             anchor: str = None) -> Tuple[List[Tuple[str, float]], str]:
        """
        Rank documents using GCCP.
        
        Args:
            query: Query text
            documents: List of candidate documents
            anchor: Optional pre-computed anchor (will generate if None)
            
        Returns:
            Tuple of (rankings, anchor) where rankings is list of (docid, score)
        """
        # Generate anchor if not provided
        if anchor is None:
            anchor = self.generate_anchor(documents)
        
        # Score each document against anchor
        scores = []
        for doc in tqdm(documents, desc="GCCP Scoring", leave=False):
            score = self._get_comparison_score(
                query, 
                doc.get('contents', doc.get('text', '')),
                anchor
            )
            scores.append((doc['docid'], score))
        
        rankings = sorted(scores, key=lambda x: x[1], reverse=True)
        return rankings, anchor
    
    def batch_rank(self, queries: Dict[str, str], 
                   all_documents: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Rank documents for multiple queries.
        
        Args:
            queries: Dict mapping query_id to query text
            all_documents: Dict mapping query_id to candidate documents
            
        Returns:
            Dict mapping query_id to rankings
        """
        results = {}
        
        for qid, query in tqdm(queries.items(), desc="GCCP Ranking"):
            documents = all_documents[qid]
            rankings, _ = self.rank(query, documents)
            results[qid] = rankings
        
        return results


if __name__ == "__main__":
    # Test GCCP
    model_name = "google/flan-t5-small"
    
    query = "what is the capital of France"
    documents = [
        {"docid": "1", "contents": "Paris is the capital and largest city of France. It is a major European city."},
        {"docid": "2", "contents": "The Eiffel Tower is a famous landmark in Paris, France."},
        {"docid": "3", "contents": "Berlin is the capital of Germany. It has a rich history."},
        {"docid": "4", "contents": "London is the capital of the United Kingdom."},
    ]
    
    print("Testing GCCP Ranker...")
    ranker = GCCPRanker(model_name, m=3, z=3, use_spacy=False)
    
    print("Generating anchor...")
    anchor = ranker.generate_anchor(documents)
    print(f"Anchor: {anchor}")
    
    print("\nRanking documents...")
    rankings, _ = ranker.rank(query, documents, anchor=anchor)
    print("Rankings:")
    for docid, score in rankings:
        print(f"  {docid}: {score:.4f}")
