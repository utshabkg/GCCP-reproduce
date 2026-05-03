"""
RG-S(0,k) ranker — relevance grading on a 0..k integer scale.

Faithful to author_code/clf_ranking.py with template_idx=4
('From a scale of 0 to {k}, judge the relevance between the query
and the document. Query: ... Document: ... Output:'). Score is peak
relevance: P(`{k}`) — author uses scores[-1].

Used by the paper's PAGC-RS-YN-GCCP and PAGC-QSG variants.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm


RG_S_TEMPLATE = (
    "From a scale of 0 to {k}, judge the relevance between the query "
    "and the document.\nQuery: {query}\nDocument: {doc_text}\nOutput:"
)


class RGSRanker:
    """T5 RG-S(0,k) ranker. Score = P(`k`) (peak relevance), normalized
    over softmax across the k+1 integer labels."""

    def __init__(
        self,
        model_name: str,
        max_doc_length: int = 128,
        scale_k: int = 4,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.scale_k = scale_k
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.float16
        )
        self.model.eval()
        # Pre-tokenize the labels and the '<pad> ' decoder primer
        self.labels = [str(i) for i in range(scale_k + 1)]
        self._label_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0] for t in self.labels
        ]
        self._decoder_input_ids = self.tokenizer.encode(
            "<pad> ", return_tensors="pt", add_special_tokens=False
        )

    def truncate(self, text: str) -> str:
        ids = self.tokenizer.tokenize(text)[: self.max_doc_length]
        return self.tokenizer.convert_tokens_to_string(ids)

    @torch.no_grad()
    def score(self, query: str, passage: str) -> float:
        prompt = RG_S_TEMPLATE.format(
            k=self.scale_k,
            query=self.truncate(query),
            doc_text=self.truncate(passage),
        )
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.to(self.model.device)
        decoder_input_ids = self._decoder_input_ids.to(self.model.device)
        out = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        logits = out.logits[0][-1]
        # Softmax over the full vocabulary, then read out the k+1 label probs
        probs = torch.softmax(logits, dim=0)
        # Peak relevance: P("k"), as in author's clf_ranking.py:
        # `relevance_score_single = scores[-1]`
        return float(probs[self._label_ids[-1]].item())

    def rank(
        self, query: str, documents: List[Dict]
    ) -> List[Tuple[str, float]]:
        scored = [
            (
                d["docid"],
                self.score(
                    query,
                    d.get("contents", d.get("text", "")),
                ),
            )
            for d in tqdm(documents, desc="RG-S Scoring", leave=False)
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored
