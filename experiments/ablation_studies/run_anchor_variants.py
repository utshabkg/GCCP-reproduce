#!/usr/bin/env python3
"""
Test plausible operationalizations of the paper's "Top" and "Random"
anchor baselines that might explain the +0.017 gap between our values
and the paper's reported BEIR aggregate (Random 0.4423 ours vs 0.4253
paper, Top-1 0.4512 vs 0.4346 paper).

Variants:
  random_5_avg : 5 random anchors {0, 42, 929, 12345, 2023}, average
                 per-query NDCG@10 across seeds
  top5_composite : top-5 sentence-interleaved composite (vs default
                   top-3)
  top1_aggressive : top-1 with 64-token truncation (vs default 128)
  top1_title : top-1 title only (no body text)

Usage:
  python experiments/ablation_studies/run_anchor_variants.py \\
      --dataset scifact --model flan-t5-large
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytrec_eval
from tqdm import tqdm

from experiments.ablation_studies.ablation_anchor import (
    DEFAULTS,
    MODEL_MAP,
    build_random_document_anchor,
    build_top1_anchor,
    build_top3_anchor,
    compute_rg_yn_results,
)
from experiments.ablation_studies.run_beir_anchor import BEIR_CONFIGS, load_beir
from src.gccp.gccp_ranker import GCCPRanker
from src.pagc.aggregation import linear_aggregation


def _avg(d, m):
    return sum(r[m] for r in d.values()) / len(d) if d else 0.0


def build_top5_composite(documents, z=10):
    """Sentence-interleaved composite over top-5 docs, capped at z sentences."""
    from experiments.ablation_studies.ablation_anchor import segment_sentences, _extract_doc_texts
    top_docs = _extract_doc_texts(documents, 5)
    if not top_docs:
        return ""
    buckets = [segment_sentences(d) for d in top_docs]
    out, idx = [], 0
    while len(out) < z:
        progress = False
        for b in buckets:
            if idx < len(b):
                out.append(b[idx])
                progress = True
                if len(out) >= z:
                    break
        if not progress:
            break
        idx += 1
    return " ".join(out[:z]) if out else " ".join(top_docs)


def build_top1_truncated(documents, max_chars=64):
    """Top-1 BM25 doc, hard-truncated to max_chars."""
    from experiments.ablation_studies.ablation_anchor import _extract_doc_texts
    docs = _extract_doc_texts(documents, 1)
    return docs[0][:max_chars] if docs else ""


def build_top1_title(documents):
    """Top-1 BM25 doc's title only (best-effort: take leading sentence
    before the first '. ', mirroring how BEIR titles are concatenated
    as 'Title. body...')."""
    if not documents:
        return ""
    text = documents[0].get("contents") or documents[0].get("text", "")
    # The Pyserini load step prepends 'Title. ' if title was present.
    if ". " in text:
        return text.split(". ", 1)[0]
    return text[:200]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(BEIR_CONFIGS.keys()))
    parser.add_argument("--model", default="flan-t5-large", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--variants",
        default="random_5_avg,top5_composite,top1_aggressive,top1_title",
        help="Comma-separated list of variants to run.",
    )
    args = parser.parse_args()

    out_path = args.output or REPO_ROOT / "results" / "ablations" / f"beir_anchor_variants_{args.dataset}_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"[{start:%H:%M:%S}] anchor variant sweep: {args.dataset} / {args.model}")

    queries, qrels, bm25 = load_beir(args.dataset)
    qids = list(queries.keys())
    if args.num_queries:
        qids = qids[: args.num_queries]
    print(f"queries: {len(qids)}")

    full_model = MODEL_MAP[args.model]
    rg_yn_results = compute_rg_yn_results(
        queries, bm25, qids, model_name=args.model, max_doc_length=DEFAULTS["max_doc_length"]
    )

    gccp = GCCPRanker(
        full_model,
        max_doc_length=DEFAULTS["max_doc_length"],
        m=DEFAULTS["m"],
        z=DEFAULTS["z"],
        threshold=DEFAULTS["threshold"],
        use_spacy=False,
    )

    test_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
    evaluator = pytrec_eval.RelevanceEvaluator(
        test_qrels, {"ndcg_cut_10", "P_10", "recall_10"}
    )

    def eval_run(gccp_results):
        bm25_run = {qid: {d["docid"]: d["score"] for d in bm25[qid][:100]} for qid in qids}
        pagc_results = {qid: linear_aggregation([rg_yn_results[qid], gccp_results[qid]]) for qid in qids}
        return {
            "bm25":  {"ndcg@10": _avg(evaluator.evaluate(bm25_run), "ndcg_cut_10")},
            "rg_yn": {"ndcg@10": _avg(evaluator.evaluate(rg_yn_results), "ndcg_cut_10")},
            "gccp":  {"ndcg@10": _avg(evaluator.evaluate(gccp_results), "ndcg_cut_10")},
            "pagc":  {"ndcg@10": _avg(evaluator.evaluate(pagc_results), "ndcg_cut_10")},
        }

    out: Dict[str, Dict] = {}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if "random_5_avg" in variants:
        # Run 5 random seeds, then average per-query NDCG@10 (the
        # statistically clean way to interpret "averaged over 5 random
        # samples"). We compute per-seed metrics first, then aggregate.
        per_seed = []
        for seed in [0, 42, 929, 12345, 2023]:
            gccp_results = {}
            for qid in tqdm(qids, desc=f"random seed={seed}"):
                docs = bm25[qid][:100]
                anchor = build_random_document_anchor(docs, qid=qid, seed=seed)
                rankings, _ = gccp.rank(queries[qid], docs, anchor=anchor)
                gccp_results[qid] = {d: s for d, s in rankings}
            per_seed.append(eval_run(gccp_results))
        # Aggregate across seeds
        agg = {}
        for k in ("bm25", "rg_yn", "gccp", "pagc"):
            mean = statistics.mean([s[k]["ndcg@10"] for s in per_seed])
            std = statistics.stdev([s[k]["ndcg@10"] for s in per_seed])
            agg[k] = {"ndcg@10": mean, "std": std}
        out["random_5_avg"] = {"per_seed": per_seed, "agg_mean_std": agg}
        print(f"  random_5_avg: gccp mean={agg['gccp']['ndcg@10']:.4f} std={agg['gccp']['std']:.4f} "
              f"pagc mean={agg['pagc']['ndcg@10']:.4f} std={agg['pagc']['std']:.4f}")

    builders_single = {
        "top5_composite":   lambda d: build_top5_composite(d, z=DEFAULTS["z"]),
        "top1_aggressive":  lambda d: build_top1_truncated(d, max_chars=64),
        "top1_title":       lambda d: build_top1_title(d),
    }
    for name in variants:
        if name == "random_5_avg":
            continue
        builder = builders_single[name]
        gccp_results = {}
        for qid in tqdm(qids, desc=name):
            docs = bm25[qid][:100]
            anchor = builder(docs)
            rankings, _ = gccp.rank(queries[qid], docs, anchor=anchor)
            gccp_results[qid] = {d: s for d, s in rankings}
        out[name] = {"metrics": eval_run(gccp_results)}
        m = out[name]["metrics"]
        print(f"  {name:18s} gccp={m['gccp']['ndcg@10']:.4f} pagc={m['pagc']['ndcg@10']:.4f}")

    payload = {
        "experiment": {
            "dataset": args.dataset,
            "model": args.model,
            "num_queries": len(qids),
            "elapsed": str(datetime.now() - start),
            "timestamp": start.isoformat(),
        },
        "anchor_variants": out,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
