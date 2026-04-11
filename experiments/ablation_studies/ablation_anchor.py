"""
Anchor construction ablations for GCCP.

This module keeps the core GCCP implementation unchanged and varies only the
anchor document supplied to the ranker. It is intended to live outside src/
per the collaborator instructions.
"""


from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytrec_eval
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gccp.gccp_ranker import GCCPRanker
from src.gccp.spectral_mds import generate_anchor_document, segment_sentences
from src.pagc.aggregation import linear_aggregation
from src.pointwise.rankers import RGYNRanker


MODEL_MAP = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl",
    "flan-ul2": "google/flan-ul2",
}

DEFAULTS = {
    "dataset": "dl19",
    "model": "flan-t5-large",
    "m": 10,
    "z": 10,
    "threshold": 0.2,
    "max_doc_length": 128,
    "seed": 929,
}


def load_dataset(dataset: str = "dl19") -> Tuple[Dict, Dict, Dict]:
    """Load queries and BM25 candidates"""
    data_dir = REPO_ROOT / "data"
    queries = json.loads((data_dir / f"{dataset}_queries.json").read_text())
    qrels = json.loads((data_dir / f"{dataset}_qrels.json").read_text())
    raw_results = json.loads((data_dir / f"{dataset}_pyserini_bm25.json").read_text())

    bm25_results = {}
    for qid, data in raw_results.items():
        bm25_results[qid] = [
            {"docid": passage["pid"], "score": 100 - idx, "contents": passage["text"]}
            for idx, passage in enumerate(data["passages"])
        ]
    return queries, qrels, bm25_results


def _extract_doc_texts(documents: List[Dict], limit: int | None = None) -> List[str]:
    docs = documents if limit is None else documents[:limit]
    return [doc.get("contents", doc.get("text", "")).strip() for doc in docs if doc]


def _stable_qid_seed(qid: str, seed: int) -> int:
    digest = hashlib.sha256(f"{qid}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_random_document_anchor(documents: List[Dict], qid: str, seed: int = 929) -> str:
    """Using random doc from top m pool as anchor"""
    candidates = _extract_doc_texts(documents, DEFAULTS["m"])
    if not candidates:
        return ""
    rng = random.Random(_stable_qid_seed(qid, seed))
    return candidates[rng.randrange(len(candidates))]


def build_top1_anchor(documents: List[Dict]) -> str:
    """Using highest ranked BM25 as anchor"""
    candidates = _extract_doc_texts(documents, 1)
    return candidates[0] if candidates else ""


def build_top3_anchor(documents: List[Dict], z: int = 10) -> str:
    """Composite anchor from top 3 BM25 docs"""
    top_docs = _extract_doc_texts(documents, 3)
    if not top_docs:
        return ""

    sentence_buckets = [segment_sentences(doc) for doc in top_docs]
    anchor_sentences = []
    sentence_idx = 0

    while len(anchor_sentences) < z:
        made_progress = False
        for bucket in sentence_buckets:
            if sentence_idx < len(bucket):
                anchor_sentences.append(bucket[sentence_idx])
                made_progress = True
                if len(anchor_sentences) >= z:
                    break
        if not made_progress:
            break
        sentence_idx += 1

    if not anchor_sentences:
        return " ".join(top_docs)
    return " ".join(anchor_sentences[:z])


def build_spectral_anchor(
    documents: List[Dict],
    m: int = 10,
    z: int = 10,
    threshold: float = 0.2,
    use_spacy: bool = False,
) -> str:
    """Use repo's default spectral MDS anchor"""
    return generate_anchor_document(
        documents,
        m=m,
        z=z,
        threshold=threshold,
        use_spacy=use_spacy,
    )


def get_anchor_builders() -> Dict[str, Callable[..., str]]:
    return {
        "random_document": build_random_document_anchor,
        "top1_bm25": build_top1_anchor,
        "top3_composite": build_top3_anchor,
        "spectral_mds": build_spectral_anchor,
    }


def _avg_metric(results: Dict[str, Dict[str, float]], metric: str) -> float:
    return sum(row[metric] for row in results.values()) / len(results)


def evaluate_runs(
    qrels: Dict[str, Dict[str, int]],
    bm25_results: Dict[str, List[Dict]],
    rg_yn_results: Dict[str, Dict[str, float]],
    gccp_results: Dict[str, Dict[str, float]],
    qids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Evaluate BM25, GCCP, and PAGC runs with pytrec_eval."""
    test_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(
        test_qrels, {"ndcg_cut_10", "P_10", "recall_10"}
    )

    pagc_results = {
        qid: linear_aggregation([rg_yn_results[qid], gccp_results[qid]])
        for qid in qids
    }
    bm25_run = {
        qid: {doc["docid"]: doc["score"] for doc in bm25_results[qid][:100]}
        for qid in qids
    }

    bm25_eval = evaluator.evaluate(bm25_run)
    rg_yn_eval = evaluator.evaluate(rg_yn_results)
    gccp_eval = evaluator.evaluate(gccp_results)
    pagc_eval = evaluator.evaluate(pagc_results)

    return {
        "bm25": {
            "ndcg@10": _avg_metric(bm25_eval, "ndcg_cut_10"),
            "p@10": _avg_metric(bm25_eval, "P_10"),
            "recall@10": _avg_metric(bm25_eval, "recall_10"),
        },
        "rg_yn": {
            "ndcg@10": _avg_metric(rg_yn_eval, "ndcg_cut_10"),
            "p@10": _avg_metric(rg_yn_eval, "P_10"),
            "recall@10": _avg_metric(rg_yn_eval, "recall_10"),
        },
        "gccp": {
            "ndcg@10": _avg_metric(gccp_eval, "ndcg_cut_10"),
            "p@10": _avg_metric(gccp_eval, "P_10"),
            "recall@10": _avg_metric(gccp_eval, "recall_10"),
        },
        "pagc": {
            "ndcg@10": _avg_metric(pagc_eval, "ndcg_cut_10"),
            "p@10": _avg_metric(pagc_eval, "P_10"),
            "recall@10": _avg_metric(pagc_eval, "recall_10"),
        },
    }


def compute_rg_yn_results(
    queries: Dict[str, str],
    bm25_results: Dict[str, List[Dict]],
    qids: List[str],
    model_name: str,
    max_doc_length: int = 128,
) -> Dict[str, Dict[str, float]]:
    """Compute and cache RG-YN scores once for reuse across ablations."""
    full_model_name = MODEL_MAP.get(model_name, model_name)
    ranker = RGYNRanker(
        full_model_name,
        template_idx=0,
        target_tokens=("yes", "no"),
        max_doc_length=max_doc_length,
    )

    results = {}
    for qid in tqdm(qids, desc="RG-YN baseline"):
        query = queries[qid]
        docs = bm25_results[qid][:100]
        rankings = ranker.rank(query, docs)
        results[qid] = {docid: score for docid, score in rankings}
    return results


def run_anchor_method_ablation(
    dataset: str = "dl19",
    model_name: str = "flan-t5-large",
    num_queries: int | None = None,
    output_path: str | os.PathLike | None = None,
    seed: int = 929,
    rg_yn_results: Dict[str, Dict[str, float]] | None = None,
) -> Dict:
    """Run the four anchor-method ablations on the specified dataset."""
    if dataset != "dl19":
        raise ValueError("Collaborator 2 ablations are scoped to dl19 by default.")

    start_time = datetime.now()
    queries, qrels, bm25_results = load_dataset(dataset)
    qids = list(queries.keys())[:num_queries] if num_queries else list(queries.keys())
    full_model_name = MODEL_MAP.get(model_name, model_name)

    if rg_yn_results is None:
        rg_yn_results = compute_rg_yn_results(
            queries,
            bm25_results,
            qids,
            model_name=model_name,
            max_doc_length=DEFAULTS["max_doc_length"],
        )

    gccp_ranker = GCCPRanker(
        full_model_name,
        max_doc_length=DEFAULTS["max_doc_length"],
        m=DEFAULTS["m"],
        z=DEFAULTS["z"],
        threshold=DEFAULTS["threshold"],
        use_spacy=False,
    )

    anchor_results = {}
    builders = get_anchor_builders()

    for method_name, builder in builders.items():
        gccp_results = {}
        for qid in tqdm(qids, desc=f"Anchor ablation: {method_name}"):
            query = queries[qid]
            docs = bm25_results[qid][:100]

            if method_name == "random_document":
                anchor = builder(docs, qid=qid, seed=seed)
            elif method_name == "top3_composite":
                anchor = builder(docs, z=DEFAULTS["z"])
            elif method_name == "spectral_mds":
                anchor = builder(
                    docs,
                    m=DEFAULTS["m"],
                    z=DEFAULTS["z"],
                    threshold=DEFAULTS["threshold"],
                    use_spacy=False,
                )
            else:
                anchor = builder(docs)

            rankings, _ = gccp_ranker.rank(query, docs, anchor=anchor)
            gccp_results[qid] = {docid: score for docid, score in rankings}

        metrics = evaluate_runs(qrels, bm25_results, rg_yn_results, gccp_results, qids)
        anchor_results[method_name] = {
            "config": {
                "anchor_method": method_name,
                "m": DEFAULTS["m"],
                "z": DEFAULTS["z"],
                "threshold": DEFAULTS["threshold"],
                "seed": seed if method_name == "random_document" else None,
            },
            "metrics": metrics,
        }

    payload = {
        "experiment": {
            "dataset": dataset,
            "model": model_name,
            "num_queries": len(qids),
            "timestamp": start_time.isoformat(),
            "elapsed": str(datetime.now() - start_time),
        },
        "anchor_methods": anchor_results,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2))

    return payload
