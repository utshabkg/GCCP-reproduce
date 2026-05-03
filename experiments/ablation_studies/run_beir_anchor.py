#!/usr/bin/env python3
"""
BEIR anchor-construction ablation, paralleling the DL19/DL20 sweep.
Reuses ablation_anchor.build_* functions and the same evaluator, but
loads the BEIR corpus + queries via pyserini's prebuilt Lucene indices
(matching scripts/run_beir.py).

Output:
    results/ablations/beir_anchor_<dataset>_t5large.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytrec_eval
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from tqdm import tqdm

from experiments.ablation_studies.ablation_anchor import (
    DEFAULTS,
    MODEL_MAP,
    build_random_document_anchor,
    build_top1_anchor,
    build_top3_anchor,
    build_spectral_anchor,
    compute_rg_yn_results,
    evaluate_runs,
)
from src.gccp.gccp_ranker import GCCPRanker

# Pyserini index/topic/qrels names per BEIR set (mirror scripts/run_beir.py)
BEIR_CONFIGS = {
    "scifact":          ("beir-v1.0.0-scifact.flat",          "beir-v1.0.0-scifact-test",          "beir-v1.0.0-scifact-test"),
    "nfcorpus":         ("beir-v1.0.0-nfcorpus.flat",         "beir-v1.0.0-nfcorpus-test",         "beir-v1.0.0-nfcorpus-test"),
    "trec-covid":       ("beir-v1.0.0-trec-covid.flat",       "beir-v1.0.0-trec-covid-test",       "beir-v1.0.0-trec-covid-test"),
    "webis-touche2020": ("beir-v1.0.0-webis-touche2020.flat", "beir-v1.0.0-webis-touche2020-test", "beir-v1.0.0-webis-touche2020-test"),
    "dbpedia-entity":   ("beir-v1.0.0-dbpedia-entity.flat",   "beir-v1.0.0-dbpedia-entity-test",   "beir-v1.0.0-dbpedia-entity-test"),
    "robust04":         ("beir-v1.0.0-robust04.flat",         "beir-v1.0.0-robust04-test",         "beir-v1.0.0-robust04-test"),
    "trec-news":        ("beir-v1.0.0-trec-news.flat",        "beir-v1.0.0-trec-news-test",        "beir-v1.0.0-trec-news-test"),
    "signal1m":         ("beir-v1.0.0-signal1m.flat",         "beir-v1.0.0-signal1m-test",         "beir-v1.0.0-signal1m-test"),
}


def load_beir(dataset: str, top_k: int = 100):
    idx, topics_name, qrels_name = BEIR_CONFIGS[dataset]
    print(f"[{datetime.now():%H:%M:%S}] loading pyserini index {idx}")
    searcher = LuceneSearcher.from_prebuilt_index(idx)
    topics = get_topics(topics_name)
    qrels = get_qrels(qrels_name)

    queries = {}
    bm25_results = {}
    for qid, ts in tqdm(topics.items(), desc=f"BM25 search ({dataset})"):
        # Pyserini get_topics returns a dict of dicts with 'title' key
        if isinstance(ts, dict):
            query = ts.get("title", "") or ts.get("description", "") or ""
        else:
            query = str(ts)
        queries[str(qid)] = query
        if not query:
            bm25_results[str(qid)] = []
            continue
        hits = searcher.search(query, k=top_k)
        docs = []
        for rank, h in enumerate(hits):
            doc = searcher.doc(h.docid)
            if doc is None:
                continue
            raw = json.loads(doc.raw())
            text = raw.get("text", "") or raw.get("contents", "")
            title = raw.get("title", "")
            if title and title not in text:
                text = f"{title}. {text}"
            docs.append({"docid": h.docid, "score": float(h.score), "contents": text})
        bm25_results[str(qid)] = docs

    # qrels keys are strings already from get_qrels; normalize
    qrels_norm = {str(qid): {str(d): int(rel) for d, rel in r.items()} for qid, r in qrels.items()}
    return queries, qrels_norm, bm25_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(BEIR_CONFIGS.keys()))
    parser.add_argument("--model", default="flan-t5-large", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=929)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.output or REPO_ROOT / "results" / "ablations" / f"beir_anchor_{args.dataset}_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] BEIR anchor ablation: {args.dataset} / {args.model}")

    queries, qrels, bm25 = load_beir(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    full_model = MODEL_MAP[args.model]

    # Compute RG-YN once and reuse (RG-YN doesn't depend on anchor)
    rg_yn_results = compute_rg_yn_results(queries, bm25, qids, model_name=args.model, max_doc_length=DEFAULTS["max_doc_length"])

    gccp = GCCPRanker(full_model, max_doc_length=DEFAULTS["max_doc_length"], m=DEFAULTS["m"], z=DEFAULTS["z"], threshold=DEFAULTS["threshold"], use_spacy=False)

    methods = {
        "random_document": lambda d, qid: build_random_document_anchor(d, qid=qid, seed=args.seed),
        "top1_bm25":       lambda d, qid: build_top1_anchor(d),
        "top3_composite":  lambda d, qid: build_top3_anchor(d, z=DEFAULTS["z"]),
        "spectral_mds":    lambda d, qid: build_spectral_anchor(d, m=DEFAULTS["m"], z=DEFAULTS["z"], threshold=DEFAULTS["threshold"], use_spacy=False),
    }

    out: Dict[str, Dict] = {}
    for name, builder in methods.items():
        gccp_results = {}
        for qid in tqdm(qids, desc=f"anchor: {name}"):
            docs = bm25[qid][:100]
            anchor = builder(docs, qid)
            rankings, _ = gccp.rank(queries[qid], docs, anchor=anchor)
            gccp_results[qid] = {d: s for d, s in rankings}
        metrics = evaluate_runs(qrels, bm25, rg_yn_results, gccp_results, qids)
        out[name] = {"config": {"anchor_method": name, "m": DEFAULTS["m"], "z": DEFAULTS["z"], "threshold": DEFAULTS["threshold"], "seed": args.seed if name == "random_document" else None},
                     "metrics": metrics}
        m = metrics
        print(f"  {name:20s}: rg_yn={m['rg_yn']['ndcg@10']:.4f} gccp={m['gccp']['ndcg@10']:.4f} pagc={m['pagc']['ndcg@10']:.4f}")

    payload = {
        "experiment": {"dataset": args.dataset, "model": args.model, "num_queries": len(qids), "elapsed": str(datetime.now() - start), "timestamp": start.isoformat()},
        "anchor_methods": out,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
