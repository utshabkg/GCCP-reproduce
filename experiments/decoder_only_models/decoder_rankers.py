"""
Decoder-only RG-YN and GCCP rankers (LLaMA-3.1, Qwen-2.5, ...).

Novel extension beyond the base paper, which only tested encoder-decoder
Flan-T5 / Flan-UL2. We adapt the same prompts to chat-template formats
and score by next-token probability over yes/no (RG-YN) or A/B (GCCP).

Differences from src/pointwise/rankers.py and src/gccp/gccp_ranker.py:
- Uses tokenizer.apply_chat_template(..., add_generation_prompt=True)
  to put the model in 'about-to-respond' state, then reads logits[-1].
- Aggregates probability mass over case/space variants of yes/no/A/B
  to be robust to tokenizer quirks (e.g. ' yes' vs 'yes').
- No T5-style decoder_input_ids (causal LM).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Same prompts as the encoder-decoder pipeline (only the wrapping changes).
RG_YN_PROMPT = (
    "Passage: {passage}\n"
    "Query: {query}\n"
    "Is the passage relevant to the query? Answer 'yes' or 'no'."
)

GCCP_PROMPT = (
    'Given a query "{query}", which of the following two passages is more relevant to the query?\n\n'
    'Passage A: "{passage_a}"\n\n'
    'Passage B: "{passage_b}"\n\n'
    'Output Passage A or Passage B:'
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful information retrieval assistant. Answer concisely "
    "with the single token requested by the user."
)


class _DecoderBase:
    """Shared infrastructure: model loading, truncation, single-token scoring."""

    def __init__(
        self,
        model_name: str,
        max_doc_length: int = 128,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_doc_length = max_doc_length
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map
        )
        self.model.eval()

    # --- text utilities ---

    def truncate(self, text: str) -> str:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids[: self.max_doc_length]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_prompt_ids(self, user_message: str, response_prefix: str = "") -> torch.Tensor:
        """Build input_ids ending right before the next token to score.

        With the GCCP prompt the model would naturally emit ``Passage A`` /
        ``Passage B`` rather than ``A`` / ``B`` directly, so callers can pass
        ``response_prefix=' Passage '`` to put the model in "I've already
        said 'Passage ', what comes next?" state. This mirrors the T5
        decoder-input convention used by the original authors.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if response_prefix:
            text = text + response_prefix
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.model.device)

    def _first_token_variants(self, words: List[str]) -> List[int]:
        """Return unique first-token IDs across a set of casings/spacings."""
        seen: List[int] = []
        seen_set = set()
        for w in words:
            ids = self.tokenizer(w, add_special_tokens=False).input_ids
            if not ids:
                continue
            tid = ids[0]
            if tid not in seen_set:
                seen.append(tid)
                seen_set.add(tid)
        return seen

    # --- core scoring ---

    @torch.no_grad()
    def _next_token_probs(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(prompt_ids)
        logits = out.logits[0, -1].float()
        return torch.softmax(logits, dim=-1)


class DecoderOnlyRGYNRanker(_DecoderBase):
    """Pointwise relevance score from P('yes') / (P('yes') + P('no'))."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self._yes_ids = self._first_token_variants(
            ["yes", " yes", "Yes", " Yes", "YES", " YES"]
        )
        self._no_ids = self._first_token_variants(
            ["no", " no", "No", " No", "NO", " NO"]
        )

    def score(self, query: str, passage: str) -> float:
        passage = self.truncate(passage)
        query = self.truncate(query)
        user = RG_YN_PROMPT.format(passage=passage, query=query)
        ids = self._build_prompt_ids(user)
        probs = self._next_token_probs(ids)
        yes_p = float(probs[self._yes_ids].sum().item())
        no_p = float(probs[self._no_ids].sum().item())
        return yes_p / (yes_p + no_p) if (yes_p + no_p) > 0 else 0.5

    def rank(self, query: str, documents: List[Dict]) -> List[Tuple[str, float]]:
        scored = [
            (d["docid"], self.score(query, d.get("contents", d.get("text", ""))))
            for d in documents
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored


class DecoderOnlyGCCPRanker(_DecoderBase):
    """Contrastive relevance score against an anchor document.

    Score for the document = P('A') normalized over P('A')+P('B'), where
    Passage A is the candidate document and Passage B is the anchor.
    """

    def __init__(
        self,
        model_name: str,
        anchor_fn=None,
        m: int = 10,
        z: int = 10,
        threshold: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(model_name, **kwargs)
        # Default anchor: in-tree spectral MDS
        if anchor_fn is None:
            from src.gccp.spectral_mds import generate_anchor_document

            def _default(documents):
                texts = [d.get("contents", d.get("text", "")) for d in documents[:m]]
                return generate_anchor_document(
                    texts, m=m, z=z, threshold=threshold, use_spacy=False
                )

            anchor_fn = _default
        self.anchor_fn = anchor_fn

        self._a_ids = self._first_token_variants(["A", " A"])
        self._b_ids = self._first_token_variants(["B", " B"])

    def _comparison_score(self, query: str, passage: str, anchor: str) -> float:
        prompt = GCCP_PROMPT.format(
            query=self.truncate(query),
            passage_a=self.truncate(passage),
            passage_b=self.truncate(anchor),
        )
        # Prime with "Passage " so we score P(A|...Passage ) vs P(B|...Passage ),
        # matching the T5 decoder_input_text='<pad> Passage ' convention.
        ids = self._build_prompt_ids(prompt, response_prefix="Passage ")
        probs = self._next_token_probs(ids)
        a_p = float(probs[self._a_ids].sum().item())
        b_p = float(probs[self._b_ids].sum().item())
        return a_p / (a_p + b_p) if (a_p + b_p) > 0 else 0.5

    def rank(
        self,
        query: str,
        documents: List[Dict],
        anchor: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float]], str]:
        if anchor is None:
            anchor = self.anchor_fn(documents)
        scored = [
            (d["docid"], self._comparison_score(query, d.get("contents", d.get("text", "")), anchor))
            for d in documents
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored, anchor
