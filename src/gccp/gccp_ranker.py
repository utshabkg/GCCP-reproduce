"""
GCCP: Global-Consistent Comparative Pointwise Ranking

Implements contrastive relevance scoring using anchor documents (Eq. 10).
Based on author's implementation: https://github.com/ChainsawM/GCCP
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

from .spectral_mds import generate_anchor_document


# Author's compare_prompt_templates[0] - exact match
GCCP_PROMPT = '''Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{passage_a}"

Passage B: "{passage_b}"

Output Passage A or Passage B:'''


class GCCPRanker:
    """
    Global-Consistent Comparative Pointwise Ranker.
    
    Uses an anchor document as reference for pairwise comparison
    within a pointwise framework.
    """
    
    def __init__(self, model_name: str, device: str = None,
                 max_doc_length: int = 128, m: int = 10, z: int = 10,
                 threshold: float = 0.2, use_spacy: bool = False,
                 sentencizer: str = 'nltk'):
        """
        Initialize GCCP ranker.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            max_doc_length: Maximum doc length in tokens (128 per author)
            m: Number of top docs for anchor generation (10 per author)
            z: Number of sentences in anchor (10 per author)
            threshold: Similarity threshold for MDS (0.2 per author)
            use_spacy: Use spaCy for sentence segmentation
            sentencizer: 'spacy' or 'nltk' (author uses both, nltk default)
        """
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.m = m
        self.z = z
        self.threshold = threshold
        self.use_spacy = use_spacy or (sentencizer == 'spacy')
        self.sentencizer = sentencizer
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model - following author's model_handler.py
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
        """Truncate text to specified token length."""
        if length is None:
            length = self.max_doc_length
        tokens = self.tokenizer.tokenize(text)[:length]
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def likelihood(self, input_text: str, target_tokens: List[str],
                   decoder_input_text: str = '<pad> Passage ') -> List[float]:
        """
        Get softmax probabilities for target tokens.
        
        Note: For GCCP, author uses decoder_input_text='<pad> Passage '
        to prime the decoder to output "Passage A" or "Passage B"
        """
        input_ids = self.tokenizer(
            input_text, return_tensors='pt', truncation=True
        ).input_ids.to(self.device)
        
        decoder_input_ids = self.tokenizer.encode(
            decoder_input_text, return_tensors='pt', add_special_tokens=False
        ).to(self.device)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                output = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                logits = output.logits[0][-1]
            else:
                output = self.model(input_ids=input_ids)
                logits = output.logits[0][-1]
        
        # Softmax probabilities (not log!)
        distributions = torch.softmax(logits, dim=0)
        
        scores = []
        for tt in target_tokens:
            token_id = self.tokenizer.encode(tt, add_special_tokens=False)[0]
            scores.append(distributions[token_id].cpu().item())
        
        return scores
    
    def _get_comparison_score(self, query: str, passage: str, anchor: str) -> float:
        """
        Compute contrastive relevance score (Eq. 10).
        
        Following author's implementation:
        - Document is Passage A, Anchor is Passage B
        - Return P(A) as relevance score (way_score='single')
        """
        # Truncate
        query_trunc = self.truncate(query)
        passage_trunc = self.truncate(passage)
        anchor_trunc = self.truncate(anchor)
        
        # Format prompt - doc first, anchor second (author's convention)
        prompt = GCCP_PROMPT.format(
            query=query_trunc,
            passage_a=passage_trunc,
            passage_b=anchor_trunc
        )
        
        # Get P(A) and P(B) with decoder primed with '<pad> Passage '
        scores = self.likelihood(prompt, ['A', 'B'], decoder_input_text='<pad> Passage ')
        
        # Return P(A) - probability passage is more relevant than anchor
        return scores[0]
    
    def generate_anchor(self, documents: List[Dict]) -> str:
        """
        Generate anchor document from top-m candidates using spectral MDS.
        
        Args:
            documents: List of documents (sorted by initial ranking)
            
        Returns:
            Anchor document text (z sentences concatenated)
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
