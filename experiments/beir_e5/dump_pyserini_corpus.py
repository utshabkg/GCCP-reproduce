#!/usr/bin/env python3
"""
Dump BEIR corpus/queries/qrels via Pyserini for datasets not in ir_datasets.

Used for trec-news, robust04, signal1m where ir_datasets has no BEIR alias.
Run in gccp-reproduce env (where pyserini is installed).

Outputs:
    data/beir_e5_pyserini_dump/<dataset>/corpus.jsonl   (one JSON per line)
    data/beir_<dataset>_queries.json
    data/beir_<dataset>_qrels.json
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

from pyserini.index.lucene import IndexReader
from pyserini.search import get_topics, get_qrels
from tqdm import tqdm

BEIR_CONFIGS = {
    "trec-news": ("beir-v1.0.0-trec-news.flat", "beir-v1.0.0-trec-news-test", "beir-v1.0.0-trec-news-test"),
    "robust04":  ("beir-v1.0.0-robust04.flat",  "beir-v1.0.0-robust04-test",  "beir-v1.0.0-robust04-test"),
    "signal1m":  ("beir-v1.0.0-signal1m.flat",  "beir-v1.0.0-signal1m-test",  "beir-v1.0.0-signal1m-test"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(BEIR_CONFIGS.keys()))
    args = parser.parse_args()

    idx_name, topics_name, qrels_name = BEIR_CONFIGS[args.dataset]
    out_dir = REPO_ROOT / "data" / "beir_e5_pyserini_dump" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = REPO_ROOT / "data" / f"beir_{args.dataset}_queries.json"
    qrels_path = REPO_ROOT / "data" / f"beir_{args.dataset}_qrels.json"

    print(f"[{datetime.now():%H:%M:%S}] opening pyserini index {idx_name}")
    reader = IndexReader.from_prebuilt_index(idx_name)
    n_docs = reader.stats()["documents"]
    print(f"  N docs = {n_docs:,}")

    print(f"[{datetime.now():%H:%M:%S}] dumping corpus -> {corpus_path}")
    with corpus_path.open("w") as fh:
        for i in tqdm(range(n_docs), desc="corpus"):
            docid = reader.convert_internal_docid_to_collection_docid(i)
            if not docid:
                continue
            raw = reader.doc_raw(docid)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                obj = {"text": raw}
            text = obj.get("text") or obj.get("contents") or ""
            title = obj.get("title", "") or ""
            fh.write(json.dumps({"id": docid, "title": title, "text": text}) + "\n")

    print(f"[{datetime.now():%H:%M:%S}] dumping topics/qrels")
    topics = get_topics(topics_name)
    qrels = get_qrels(qrels_name)

    queries_dict = {}
    for qid, ts in topics.items():
        if isinstance(ts, dict):
            q = ts.get("title") or ts.get("description") or ""
        else:
            q = str(ts)
        queries_dict[str(qid)] = q

    qrels_dict = {}
    for qid, docrels in qrels.items():
        qrels_dict[str(qid)] = {str(d): int(r) for d, r in docrels.items()}

    queries_path.write_text(json.dumps(queries_dict))
    qrels_path.write_text(json.dumps(qrels_dict))
    print(f"  wrote {len(queries_dict)} queries, {len(qrels_dict)} qrels")
    print(f"Saved corpus:  {corpus_path}")
    print(f"Saved queries: {queries_path}")
    print(f"Saved qrels:   {qrels_path}")


if __name__ == "__main__":
    main()
