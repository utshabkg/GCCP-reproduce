"""
Pointwise Ranking Methods: RG-YN, RG-S, QG

Implements the baseline pointwise scoring approaches:
- RG-YN: Relevance Generation with Yes/No
- RG-S(0,k): Relevance Generation with Scale 0-k
- QG: Query Generation likelihood
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


# Prompt templates following the original paper
PROMPT_TEMPLATES = {
    'rg_yn': """Passage: {passage}
Query: {query}
Does the passage answer the query? Answer 'Yes' or 'No'.""",

    'rg_s': """Passage: {passage}
Query: {query}
How relevant is this passage to the query? Rate from 0 (not relevant) to {max_score} (highly relevant).""",

    'qg': """Passage: {passage}
Please write a question based on this passage.""",
}


class PointwiseRanker:
    """Base class for pointwise ranking methods."""
    
    def __init__(self, model_name: str, device: str = None, 
                 max_length: int = 512, batch_size: int = 8):
        """
        Initialize the pointwise ranker.
        
        Args:
            model_name: HuggingFace model name (e.g., 'google/flan-t5-xl')
            device: Device to use (cuda/cpu)
            max_length: Maximum input length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if encoder-decoder or decoder-only
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
    
    def _get_token_logits(self, input_text: str, target_tokens: List[str]) -> Dict[str, float]:
        """Get log probabilities for specific target tokens."""
        inputs = self.tokenizer(
            input_text, return_tensors='pt', 
            max_length=self.max_length, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                # For encoder-decoder models, we need decoder input
                decoder_input = self.tokenizer(
                    "", return_tensors='pt'
                ).input_ids.to(self.device)
                
                outputs = self.model(
                    **inputs, 
                    decoder_input_ids=decoder_input
                )
                logits = outputs.logits[0, -1, :]  # Last position logits
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
        
        # Get log probabilities for target tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        results = {}
        for token in target_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)[0]
            results[token] = log_probs[token_id].item()
        
        return results


class RGYNRanker(PointwiseRanker):
    """Relevance Generation with Yes/No scoring (Eq. 3 in paper)."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.template = PROMPT_TEMPLATES['rg_yn']
    
    def score(self, query: str, passage: str) -> float:
        """
        Score a single query-passage pair.
        
        f_RG-YN(q, d) = exp(S_Y) / (exp(S_Y) + exp(S_N))
        
        Returns probability of 'Yes' (relevance score)
        """
        prompt = self.template.format(query=query, passage=passage)
        log_probs = self._get_token_logits(prompt, ['Yes', 'No'])
        
        # Convert to probability (Eq. 3)
        s_yes = log_probs['Yes']
        s_no = log_probs['No']
        
        # Softmax between Yes and No
        score = np.exp(s_yes) / (np.exp(s_yes) + np.exp(s_no))
        return score
    
    def rank(self, query: str, documents: List[Dict]) -> List[Tuple[str, float]]:
        """
        Rank documents for a query.
        
        Args:
            query: Query text
            documents: List of dicts with 'docid' and 'contents'
            
        Returns:
            List of (docid, score) sorted by score descending
        """
        scores = []
        for doc in tqdm(documents, desc="RG-YN Scoring", leave=False):
            score = self.score(query, doc['contents'])
            scores.append((doc['docid'], score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)


class RGSRanker(PointwiseRanker):
    """Relevance Generation with Scale scoring (Eq. 2 in paper)."""
    
    def __init__(self, model_name: str, max_score: int = 4, **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_score = max_score
        self.template = PROMPT_TEMPLATES['rg_s']
        self.labels = [str(i) for i in range(max_score + 1)]
    
    def score(self, query: str, passage: str) -> float:
        """
        Score using Expected Relevance (Eq. 2) or Peak Relevance (Eq. 4).
        
        Using Peak Relevance (PR) as recommended in the paper:
        f_RG(q, d) = s_{i,k*} where k* is highest relevance label
        """
        prompt = self.template.format(
            query=query, passage=passage, max_score=self.max_score
        )
        log_probs = self._get_token_logits(prompt, self.labels)
        
        # Peak Relevance: use probability of highest label
        # (Paper notes PR and ER yield nearly identical results)
        return log_probs[str(self.max_score)]
    
    def score_er(self, query: str, passage: str) -> float:
        """Score using Expected Relevance (Eq. 2)."""
        prompt = self.template.format(
            query=query, passage=passage, max_score=self.max_score
        )
        log_probs = self._get_token_logits(prompt, self.labels)
        
        # Expected Relevance
        scores = np.array([log_probs[str(k)] for k in range(self.max_score + 1)])
        probs = np.exp(scores) / np.sum(np.exp(scores))
        er = sum(probs[k] * k for k in range(self.max_score + 1))
        return er
    
    def rank(self, query: str, documents: List[Dict]) -> List[Tuple[str, float]]:
        """Rank documents for a query."""
        scores = []
        for doc in tqdm(documents, desc="RG-S Scoring", leave=False):
            score = self.score(query, doc['contents'])
            scores.append((doc['docid'], score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)


class QGRanker(PointwiseRanker):
    """Query Generation scoring (Eq. 1 in paper)."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.template = PROMPT_TEMPLATES['qg']
    
    def score(self, query: str, passage: str) -> float:
        """
        Score using Query Generation likelihood (Eq. 1).
        
        f_QG(q, d) = (1/|q|) * sum_j LLM(q_j | q_{<j}, d, P_QG)
        
        Returns average log-likelihood of generating the query.
        """
        # Create prompt with passage
        prompt = self.template.format(passage=passage)
        
        inputs = self.tokenizer(
            prompt, return_tensors='pt',
            max_length=self.max_length, truncation=True
        ).to(self.device)
        
        # Tokenize query as target
        query_tokens = self.tokenizer(
            query, return_tensors='pt', add_special_tokens=False
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                outputs = self.model(
                    **inputs,
                    labels=query_tokens
                )
                # Average negative log-likelihood
                loss = outputs.loss.item()
                return -loss  # Higher is better
            else:
                # For decoder-only, compute log prob of query continuation
                full_input = self.tokenizer(
                    prompt + " " + query, return_tensors='pt',
                    max_length=self.max_length, truncation=True
                ).to(self.device)
                
                outputs = self.model(**full_input)
                logits = outputs.logits
                
                # Get log probs for query tokens
                query_start = inputs.input_ids.shape[1]
                log_probs = F.log_softmax(logits[0, query_start-1:-1], dim=-1)
                
                query_token_ids = full_input.input_ids[0, query_start:]
                token_log_probs = log_probs.gather(1, query_token_ids.unsqueeze(1))
                
                return token_log_probs.mean().item()
    
    def rank(self, query: str, documents: List[Dict]) -> List[Tuple[str, float]]:
        """Rank documents for a query."""
        scores = []
        for doc in tqdm(documents, desc="QG Scoring", leave=False):
            score = self.score(query, doc['contents'])
            scores.append((doc['docid'], score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)


def get_ranker(method: str, model_name: str, **kwargs) -> PointwiseRanker:
    """Factory function to get a ranker by method name."""
    rankers = {
        'rg_yn': RGYNRanker,
        'rg_s': RGSRanker,
        'qg': QGRanker,
    }
    
    if method not in rankers:
        raise ValueError(f"Unknown method: {method}. Available: {list(rankers.keys())}")
    
    return rankers[method](model_name, **kwargs)


if __name__ == "__main__":
    # Test with small model
    model_name = "google/flan-t5-small"
    
    query = "what is the capital of France"
    passages = [
        {"docid": "1", "contents": "Paris is the capital and largest city of France."},
        {"docid": "2", "contents": "The Eiffel Tower is located in Paris."},
        {"docid": "3", "contents": "Berlin is the capital of Germany."},
    ]
    
    print("Testing RG-YN Ranker...")
    ranker = RGYNRanker(model_name)
    rankings = ranker.rank(query, passages)
    print("Rankings:", rankings)
