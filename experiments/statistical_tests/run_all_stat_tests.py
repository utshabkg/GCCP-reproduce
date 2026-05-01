#!/usr/bin/env python3
"""
Run paired-bootstrap significance tests across all available score sets.

Discovers any directory under results/ that contains rg_yn_scores.json AND
gccp_scores.json (and optionally pagc_scores.json) and runs the standard
comparison battery: PAGC vs RG-YN, PAGC vs GCCP, GCCP vs RG-YN.

Usage:
    python experiments/statistical_tests/run_all_stat_tests.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytrec_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pagc.aggregation import linear_aggregation


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    return json.loads(Path(path).read_text())


def per_q_metric(qrels, scores, metric):
    common = {qid: qrels[qid] for qid in scores if qid in qrels}
    ev = pytrec_eval.RelevanceEvaluator(common, {metric})
    return {qid: r[metric] for qid, r in ev.evaluate({q: scores[q] for q in common}).items()}


def paired_bootstrap(base, sysrun, n=1000, seed=929):
    rng = np.random.default_rng(seed)
    diffs = sysrun - base
    delta = float(diffs.mean())
    idx = rng.integers(0, base.size, size=(n, base.size))
    bs = diffs[idx].mean(axis=1)
    if delta >= 0:
        p = 2.0 * float(np.mean(bs <= 0))
    else:
        p = 2.0 * float(np.mean(bs >= 0))
    p = min(1.0, p)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return delta, p, float(lo), float(hi)


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def discover_score_dirs() -> List[Tuple[str, Path]]:
    """Find any directory containing rg_yn_scores.json + gccp_scores.json.

    Returns list of (label, dir) sorted for deterministic output.
    """
    out = []
    for rg_path in sorted(REPO_ROOT.glob("results/**/rg_yn_scores.json")):
        d = rg_path.parent
        if (d / "gccp_scores.json").exists():
            rel = d.relative_to(REPO_ROOT / "results")
            out.append((str(rel), d))
    return out


# Map a score-dir path to the right qrels file (dl19/dl20/BEIR)
def infer_qrels(scores_dir: Path) -> Path:
    rel = scores_dir.relative_to(REPO_ROOT / "results").as_posix()
    if rel.startswith("trec-dl/dl19"):
        return REPO_ROOT / "data" / "dl19_qrels.json"
    if rel.startswith("trec-dl/dl20"):
        return REPO_ROOT / "data" / "dl20_qrels.json"
    m = re.match(r"beir/([^/]+)/", rel)
    if m:
        # BEIR qrels often have integer string keys; we don't always have them
        # cached locally — skip BEIR for now if no qrels file is available.
        return REPO_ROOT / "data" / f"beir_{m.group(1)}_qrels.json"
    return Path()


def evaluate_dir(scores_dir: Path, qrels_path: Path):
    if not qrels_path.exists():
        return None
    qrels = load_scores(qrels_path)
    rg = load_scores(scores_dir / "rg_yn_scores.json")
    gc = load_scores(scores_dir / "gccp_scores.json")

    # Reconstruct PAGC if not saved (deterministic)
    pagc_path = scores_dir / "pagc_scores.json"
    if pagc_path.exists():
        pa = load_scores(pagc_path)
    else:
        pa = {q: linear_aggregation([rg[q], gc[q]]) for q in set(rg) & set(gc)}

    metric = "ndcg_cut_10"
    rg_pq = per_q_metric(qrels, rg, metric)
    gc_pq = per_q_metric(qrels, gc, metric)
    pa_pq = per_q_metric(qrels, pa, metric)

    common = sorted(set(rg_pq) & set(gc_pq) & set(pa_pq))
    rg_arr = np.array([rg_pq[q] for q in common])
    gc_arr = np.array([gc_pq[q] for q in common])
    pa_arr = np.array([pa_pq[q] for q in common])

    out = {"n_queries": len(common), "comparisons": {}}
    for name, base, sysr in [
        ("PAGC vs RG-YN", rg_arr, pa_arr),
        ("PAGC vs GCCP", gc_arr, pa_arr),
        ("GCCP vs RG-YN", rg_arr, gc_arr),
    ]:
        delta, p, lo, hi = paired_bootstrap(base, sysr)
        n_better = int((sysr > base).sum())
        n_worse = int((sysr < base).sum())
        out["comparisons"][name] = {
            "baseline_mean": float(base.mean()),
            "system_mean": float(sysr.mean()),
            "delta_mean": delta,
            "p_value": p,
            "ci_95_lo": lo,
            "ci_95_hi": hi,
            "n_better": n_better,
            "n_worse": n_worse,
            "n_tied": len(common) - n_better - n_worse,
            "significance": sig_marker(p),
        }
    return out


def main() -> None:
    out_dir = REPO_ROOT / "results" / "stat_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict] = {}
    for label, scores_dir in discover_score_dirs():
        qrels = infer_qrels(scores_dir)
        if not qrels.exists():
            print(f"[skip] {label}: no qrels file at {qrels}")
            continue
        result = evaluate_dir(scores_dir, qrels)
        if result is None:
            continue
        summary[label] = result

        print(f"\n=== {label} ({result['n_queries']} queries) ===")
        print(f"{'Comparison':<18} {'Δ NDCG@10':>10}  {'95% CI':<22}  {'p':>8}  sig")
        for name, c in result["comparisons"].items():
            ci = f"[{c['ci_95_lo']:+.4f}, {c['ci_95_hi']:+.4f}]"
            print(
                f"{name:<18} {c['delta_mean']:>+10.4f}  {ci:<22}  "
                f"{c['p_value']:>8.4f}  {c['significance']}"
            )

    out_path = out_dir / "all_paired_bootstrap.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
