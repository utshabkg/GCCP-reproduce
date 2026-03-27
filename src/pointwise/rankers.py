"""
Pointwise Ranking Methods: RG-YN, RG-S, QG

Implements the baseline pointwise scoring approaches:
- RG-YN: Relevance Generation with Yes/No
- RG-S(0,k): Relevance Generation with Scale 0-k
- QG: Query Generation likelihood

Based on author's implementation: https://github.com/ChainsawM/GCCP
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np


# Prompt templates following the original paper (template_idx=0)
# NOTE: Using lowercase 'yes'/'no' as per author's implementation
PROMPT_TEMPLATES = {
    'rg_yn': "Passage: {passage}\nQuery: {query}\nIs the passage relevant to the query? Answer '{token_yes}' or '{token_no}'",
    
    'rg_yn_alt': "Query: {query}\nPassage: {passage}\nIs the passage relevant to the query? Answer '{token_yes}' or '{token_no}'",
    
    'rg_s': "From a scale of 0 to {k}, judge the relevance between the query and the passage.\nQuery: {query}\nPassage: {passage}\nOutput:",

    'qg': """Passage: {passage}
Please write a question based on this passage.""",
}


class PointwiseRanker:
    """Base class for pointwise ranking methods."""
    
    def __init__(self, model_name: str, device: str = None, 
                 max_doc_length: int = 128, batch_size: int = 8):
        """
        Initialize the pointwise ranker.
        
        Args:
            model_name: HuggingFace model name (e.g., 'google/flan-t5-xl')
            device: Device to use (cuda/cpu)
            max_doc_length: Maximum document length in tokens (default 128 per paper)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.batch_size = batch_size
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model and tokenizer - following author's implementation
        if 't5' in model_name.lower() or 'ul2' in model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, 
                device_map='auto',
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.is_encoder_decoder = True
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map='auto'
            )
            self.is_encoder_decoder = False
        
        self.model.eval()
    
    def truncate(self, text: str, length: int = None) -> str:
        """Truncate text to specified token length (following author's implementation)."""
        if length is None:
            length = self.max_doc_length
        tokens = self.tokenizer.tokenize(text)[:length]
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def likelihood(self, input_text: str, target_tokens: List[str], 
                   decoder_input_text: str = '<pad> ') -> List[float]:
        """
        Get softmax probabilities for target tokens (following author's implementation).
        
        This matches author's ModelHandler.likelihood() method exactly.
        """
        input_ids = self.tokenizer(
            input_text, return_tensors='pt', truncation=True
        ).input_ids.to(self.device)
        
        # For T5: use decoder_input_ids with specified prefix
        decoder_input_ids = self.tokenizer.encode(
            decoder_input_text, return_tensors='pt', add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                logits = output.logits[0][-1]  # Last position logits
            else:
                output = self.model(input_ids=input_ids)
                logits = output.logits[0][-1]
        
        # Apply softmax to get probabilities (NOT log_softmax!)
        distributions = torch.softmax(logits, dim=0)
        
        # Get probabilities for target tokens
        scores = []
        for tt in target_tokens:
            token_id = self.tokenizer.encode(tt, add_special_tokens=False)[0]
            scores.append(distributions[token_id].cpu().item())
        
        return scores


class RGYNRanker(PointwiseRanker):
    """Relevance Generation with Yes/No scoring (Eq. 3 in paper)."""
    
    def __init__(self, model_name: str, template_idx: int = 0, 
                 target_tokens: Tuple[str, str] = ('yes', 'no'), **kwargs):
        super().__init__(model_name, **kwargs)
        self.template_idx = template_idx
        self.target_tokens = target_tokens
        
        # Template selection matching author's clf_ranking.py
        if template_idx == 0:
            self.template = PROMPT_TEMPLATES['rg_yn']
        elif template_idx == 1:
            self.template = PROMPT_TEMPLATES['rg_yn_alt']
        else:
            self.template = PROMPT_TEMPLATES['rg_yn']
    
    def score(self, query: str, passage: str) -> float:
        """
        Score a single query-passage pair.
        
        Following author's implementation:
        - Truncate query and passage to max_doc_length tokens
        - Use 'yes'/'no' tokens (lowercase)
        - Return P(yes) as the relevance score
        """
        # Truncate like author does
        query_trunc = self.truncate(query)
        passage_trunc = self.truncate(passage)
        
        prompt = self.template.format(
            query=query_trunc, 
            passage=passage_trunc,
            token_yes=self.target_tokens[0],
            token_no=self.target_tokens[1]
        )
        
        # Get softmax probabilities (not log probs!)
        scores = self.likelihood(prompt, list(self.target_tokens))
        
        # Return P(yes) as relevance score
        return scores[0]
    
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
