#!/usr/bin/env python3
"""
Generate top-100 E5 retrieval results for a BEIR dataset (test split).

Run in the gccp-decoder env (sentence-transformers + faiss + ir_datasets).

Output format matches Christopher's TREC-DL E5 results so the existing
reranking flow can be reused:
    {qid: {"query": ..., "passages": [{"pid": ..., "text": ..., "score": ...}, ...]}}

Usage:
    python experiments/beir_e5/generate_beir_e5.py --dataset scifact
    python experiments/beir_e5/generate_beir_e5.py --dataset nfcorpus
    python experiments/beir_e5/generate_beir_e5.py --dataset trec-covid
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import faiss
import ir_datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Maps short BEIR dataset name -> ir_datasets identifier for the test split.
# Mirrors the eight datasets used in the original GCCP paper, although here
# we focus on a subset for the extension.
IR_DATASETS_NAMES = {
    "scifact": "beir/scifact/test",
    "nfcorpus": "beir/nfcorpus/test",
    "trec-covid": "beir/trec-covid",
    "webis-touche2020": "beir/webis-touche2020/v2",
    "dbpedia-entity": "beir/dbpedia-entity/test",
    # trec-news, robust04, signal1m are NOT in ir_datasets' BEIR namespace
    # (license/data-acquisition restrictions). For those, dump corpus via
    # Pyserini first (see dump_pyserini_corpus.py) and pass --from_dump.
}


def _load_from_dump(dataset: str):
    """Load corpus/queries/qrels from the Pyserini dump produced by
    dump_pyserini_corpus.py."""
    dump_dir = REPO_ROOT / "data" / "beir_e5_pyserini_dump" / dataset
    corpus_path = dump_dir / "corpus.jsonl"
    queries_path = REPO_ROOT / "data" / f"beir_{dataset}_queries.json"
    qrels_path = REPO_ROOT / "data" / f"beir_{dataset}_qrels.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No Pyserini dump at {corpus_path}. "
            f"Run dump_pyserini_corpus.py --dataset {dataset} first."
        )
    queries = json.loads(queries_path.read_text())
    qrels = json.loads(qrels_path.read_text())

    doc_ids = []
    doc_texts = []
    with corpus_path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            doc_ids.append(obj["id"])
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            if title and title not in text:
                doc_texts.append(f"{title}. {text}")
            else:
                doc_texts.append(text)
    return doc_ids, doc_texts, queries, qrels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="BEIR short name. ir_datasets-backed: scifact/nfcorpus/trec-covid/"
             "webis-touche2020/dbpedia-entity. Pyserini-dump: trec-news/robust04/signal1m.",
    )
    parser.add_argument(
        "--model", default="intfloat/e5-base-v2", help="E5 SentenceTransformer model"
    )
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=None, help="Default: data/beir_e5_<dataset>.json"
    )
    parser.add_argument(
        "--from_dump",
        action="store_true",
        help="Load corpus/queries/qrels from the local Pyserini dump instead of "
             "ir_datasets (required for trec-news/robust04/signal1m).",
    )
    args = parser.parse_args()

    out_path = args.output or REPO_ROOT / "data" / f"beir_e5_{args.dataset}.json"
    qrels_out = REPO_ROOT / "data" / f"beir_{args.dataset}_qrels.json"
    queries_out = REPO_ROOT / "data" / f"beir_{args.dataset}_queries.json"

    use_dump = args.from_dump or args.dataset not in IR_DATASETS_NAMES
    if use_dump:
        print(f"[{datetime.now():%H:%M:%S}] loading from Pyserini dump for {args.dataset}")
        doc_ids, doc_texts, queries_in, qrels_in = _load_from_dump(args.dataset)
        n_docs = len(doc_ids)
        n_q = len(queries_in)
        print(f"  docs={n_docs}  queries={n_q}")
        ds = None
    else:
        ds_name = IR_DATASETS_NAMES[args.dataset]
        print(f"[{datetime.now():%H:%M:%S}] loading {ds_name}")
        ds = ir_datasets.load(ds_name)
        n_docs = ds.docs_count()
        n_q = ds.queries_count()
        print(f"  docs={n_docs}  queries={n_q}")

    print(f"[{datetime.now():%H:%M:%S}] loading E5 model {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    # ------------------------------------------------------------------
    # Encode corpus
    # ------------------------------------------------------------------
    if not use_dump:
        print(f"[{datetime.now():%H:%M:%S}] reading corpus")
        docs = list(ds.docs_iter())
        doc_ids = [d.doc_id for d in docs]
        # BEIR doc namedtuples have .text and sometimes .title; concatenate.
        doc_texts = []
        for d in docs:
            title = getattr(d, "title", "") or ""
            text = getattr(d, "text", "") or ""
            if title and title not in text:
                doc_texts.append(f"{title}. {text}")
            else:
                doc_texts.append(text)
    # else: doc_ids, doc_texts already populated by _load_from_dump

    print(f"[{datetime.now():%H:%M:%S}] encoding {len(doc_texts)} docs (E5 'passage:')")
    doc_embs = model.encode(
        [f"passage: {t}" for t in doc_texts],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ------------------------------------------------------------------
    # Encode queries
    # ------------------------------------------------------------------
    print(f"[{datetime.now():%H:%M:%S}] reading queries")
    if use_dump:
        qids = list(queries_in.keys())
        q_texts = [queries_in[q] for q in qids]
    else:
        queries = list(ds.queries_iter())
        qids = [q.query_id for q in queries]
        q_texts = []
        for q in queries:
            text = getattr(q, "text", None) or getattr(q, "title", None) or getattr(q, "description", None) or ""
            q_texts.append(text)

    print(f"[{datetime.now():%H:%M:%S}] encoding {len(q_texts)} queries (E5 'query:')")
    q_embs = model.encode(
        [f"query: {t}" for t in q_texts],
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # ------------------------------------------------------------------
    # FAISS search
    # ------------------------------------------------------------------
    dim = doc_embs.shape[1]
    print(f"[{datetime.now():%H:%M:%S}] building FAISS index (dim={dim})")
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)

    print(f"[{datetime.now():%H:%M:%S}] searching top-{args.top_k}")
    scores, idxs = index.search(q_embs, args.top_k)

    # ------------------------------------------------------------------
    # Save retrieval results in the same format as the TREC-DL E5 files
    # ------------------------------------------------------------------
    out: dict = {}
    qrels_dict: dict = {}
    queries_dict: dict = {}

    if use_dump:
        for qid, dr in qrels_in.items():
            qrels_dict[qid] = dict(dr)
    else:
        qrels_iter = list(ds.qrels_iter())
        for q in qrels_iter:
            qrels_dict.setdefault(q.query_id, {})[q.doc_id] = int(q.relevance)

    for i, qid in enumerate(qids):
        passages = []
        for rank in range(args.top_k):
            j = int(idxs[i, rank])
            if j < 0:
                continue
            passages.append(
                {
                    "pid": doc_ids[j],
                    "text": doc_texts[j],
                    "score": float(scores[i, rank]),
                }
            )
        out[qid] = {"query": q_texts[i], "passages": passages}
        queries_dict[qid] = q_texts[i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    qrels_out.write_text(json.dumps(qrels_dict))
    queries_out.write_text(json.dumps(queries_dict))

    # First-stage E5 NDCG@10 sanity check
    import pytrec_eval

    common_q = {qid: qrels_dict[qid] for qid in out if qid in qrels_dict}
    e5_run = {
        qid: {p["pid"]: p["score"] for p in out[qid]["passages"]}
        for qid in common_q
    }
    ev = pytrec_eval.RelevanceEvaluator(common_q, {"ndcg_cut_10"})
    per_q = ev.evaluate(e5_run)
    if per_q:
        ndcg10 = sum(r["ndcg_cut_10"] for r in per_q.values()) / len(per_q)
        print(
            f"[{datetime.now():%H:%M:%S}] E5 first-stage NDCG@10 = {ndcg10:.4f}  "
            f"(n_eval={len(per_q)})"
        )
    print(f"Saved retrieval: {out_path}")
    print(f"Saved qrels:     {qrels_out}")
    print(f"Saved queries:   {queries_out}")


if __name__ == "__main__":
    main()
