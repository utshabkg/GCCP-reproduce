"""
Faithful port of the author's MultiDocSummarizer.extract_sentences +
spectral_summarize, used to test whether the residual DL20-T5-Large gap
is explained by sentence-segmentation differences (NLTK vs regex,
hybrid 200-char/128-char per-doc length cap, abs-Fiedler minority-side
selection).

Direct comparison source: author_code/src/modules/MultiDocSummarizer.py
(the version we cloned and audited on 2026-03-26).
"""
from __future__ import annotations

import re
from typing import List

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Make sure punkt is available; we don't fail loudly if it's missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


def _split_to_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)


def _extract_sentences_author(
    documents: List[str], max_doc_length: int = 200, min_words: int = 3
) -> List[str]:
    """Mirrors MultiDocSummarizer.extract_sentences: NLTK + dedup + hybrid 200/128 cap."""
    seen = set()
    out: List[str] = []
    for doc in documents:
        current_doc_length = 0
        for sent in _split_to_sentences(doc):
            sent = re.sub(r"\s+", " ", sent).strip()
            if len(sent.split()) >= min_words and sent not in seen:
                # Hybrid: include if total <= 200 chars OR if we still have
                # less than 128 chars in this doc (lets a long sentence in
                # even if it would push past 200 chars).
                if (
                    current_doc_length + len(sent) <= max_doc_length
                    or current_doc_length < 128
                ):
                    out.append(sent)
                    seen.add(sent)
                    current_doc_length += len(sent)
    return out


def _spectral_summarize_author(
    sentences: List[str], num_sentences: int = 10, threshold: float = 0.2
) -> List[str]:
    """Mirrors MultiDocSummarizer.spectral_summarize: abs-Fiedler minority-side selection."""
    if len(sentences) <= 1:
        return sentences
    if len(sentences) <= num_sentences:
        return sentences[:num_sentences]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)
    W = cosine_similarity(tfidf, tfidf)
    W[W < threshold] = 0  # diagonal kept (1.0)

    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-12)))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]

    scored = [(abs(score), idx, sent) for idx, (score, sent) in enumerate(zip(fiedler, sentences))]

    pos_count = int(np.sum(fiedler > 0))
    neg_count = int(np.sum(fiedler < 0))
    if pos_count < neg_count:
        scored.sort(reverse=False)
    else:
        scored.sort(reverse=True)

    selected_indices = [idx for _, idx, _ in scored[:num_sentences]]
    selected_indices.sort()
    return [sentences[i] for i in selected_indices]


def author_exact_anchor(
    documents: List[dict],
    m: int = 10,
    z: int = 10,
    threshold: float = 0.2,
    max_doc_length: int = 200,
    min_words: int = 3,
) -> str:
    """Drop-in replacement for src.gccp.spectral_mds.generate_anchor_document.

    Args:
        documents: List of {'contents'/'text': ...} dicts (top-m of BM25 list).
        m, z, threshold: paper hyperparameters (matches our defaults).
        max_doc_length, min_words: author-specific filtering knobs.
    """
    top_docs = documents[:m]
    if not top_docs:
        return ""

    if isinstance(top_docs[0], dict):
        doc_texts = [d.get("contents", d.get("text", "")) for d in top_docs]
    else:
        doc_texts = top_docs

    sentences = _extract_sentences_author(
        doc_texts, max_doc_length=max_doc_length, min_words=min_words
    )
    if not sentences:
        return ""
    summary = _spectral_summarize_author(sentences, num_sentences=z, threshold=threshold)
    return " ".join(summary)


if __name__ == "__main__":
    docs = [
        {"contents": "Paris is the capital of France. It is known for the Eiffel Tower. The city has many museums."},
        {"contents": "France is a country in Western Europe. Paris is its capital city. French cuisine is world-famous."},
        {"contents": "The Louvre Museum is located in Paris. It houses the Mona Lisa. Millions visit annually."},
    ]
    print(author_exact_anchor(docs, m=3, z=3, threshold=0.1))
