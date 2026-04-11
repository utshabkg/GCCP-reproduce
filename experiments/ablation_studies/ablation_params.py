"""
Parameter sensitivity experiments for GCCP anchor generation.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gccp.gccp_ranker import GCCPRanker

from experiments.ablation_studies.ablation_anchor import (
    DEFAULTS,
    MODEL_MAP,
    compute_rg_yn_results,
    evaluate_runs,
    load_dataset,
)


def _run_single_setting(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    bm25_results: Dict[str, List[Dict]],
    qids: List[str],
    gccp_ranker: GCCPRanker,
    rg_yn_results: Dict[str, Dict[str, float]],
    desc: str,
) -> Dict[str, Dict[str, float]]:
    gccp_results = {}
    for qid in tqdm(qids, desc=desc):
        query = queries[qid]
        docs = bm25_results[qid][:100]
        rankings, _ = gccp_ranker.rank(query, docs)
        gccp_results[qid] = {docid: score for docid, score in rankings}

    return evaluate_runs(qrels, bm25_results, rg_yn_results, gccp_results, qids)


def run_parameter_sensitivity(
    dataset: str = "dl19",
    model_name: str = "flan-t5-large",
    num_queries: int | None = None,
    output_dir: str | os.PathLike | None = None,
    m_values: List[int] | None = None,
    z_values: List[int] | None = None,
    theta_values: List[float] | None = None,
    rg_yn_results: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict]:
    """Run sensitivity sweeps for m, z, and theta."""
    if dataset != "dl19":
        raise ValueError("Collaborator 2 ablations are scoped to dl19 by default.")

    start_time = datetime.now()
    queries, qrels, bm25_results = load_dataset(dataset)
    qids = list(queries.keys())[:num_queries] if num_queries else list(queries.keys())
    full_model_name = MODEL_MAP.get(model_name, model_name)

    m_values = m_values or [5, 10, 15, 20]
    z_values = z_values or [5, 10, 15, 20]
    theta_values = theta_values or [0.1, 0.2, 0.3, 0.4]

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

    sensitivity_results = {}

    m_runs = []
    for value in m_values:
        gccp_ranker.m = value
        gccp_ranker.z = DEFAULTS["z"]
        gccp_ranker.threshold = DEFAULTS["threshold"]
        metrics = _run_single_setting(
            queries,
            qrels,
            bm25_results,
            qids,
            gccp_ranker,
            rg_yn_results,
            desc=f"m sensitivity: {value}",
        )
        m_runs.append(
            {
                "setting": value,
                "fixed": {"z": DEFAULTS["z"], "threshold": DEFAULTS["threshold"]},
                "metrics": metrics,
            }
        )
    sensitivity_results["m"] = m_runs

    z_runs = []
    for value in z_values:
        gccp_ranker.m = DEFAULTS["m"]
        gccp_ranker.z = value
        gccp_ranker.threshold = DEFAULTS["threshold"]
        metrics = _run_single_setting(
            queries,
            qrels,
            bm25_results,
            qids,
            gccp_ranker,
            rg_yn_results,
            desc=f"z sensitivity: {value}",
        )
        z_runs.append(
            {
                "setting": value,
                "fixed": {"m": DEFAULTS["m"], "threshold": DEFAULTS["threshold"]},
                "metrics": metrics,
            }
        )
    sensitivity_results["z"] = z_runs

    theta_runs = []
    for value in theta_values:
        gccp_ranker.m = DEFAULTS["m"]
        gccp_ranker.z = DEFAULTS["z"]
        gccp_ranker.threshold = value
        metrics = _run_single_setting(
            queries,
            qrels,
            bm25_results,
            qids,
            gccp_ranker,
            rg_yn_results,
            desc=f"theta sensitivity: {value}",
        )
        theta_runs.append(
            {
                "setting": value,
                "fixed": {"m": DEFAULTS["m"], "z": DEFAULTS["z"]},
                "metrics": metrics,
            }
        )
    sensitivity_results["theta"] = theta_runs

    payload = {
        "experiment": {
            "dataset": dataset,
            "model": model_name,
            "num_queries": len(qids),
            "timestamp": start_time.isoformat(),
            "elapsed": str(datetime.now() - start_time),
        },
        "sensitivity": sensitivity_results,
    }

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "param_m_sensitivity.json").write_text(
            json.dumps({"experiment": payload["experiment"], "parameter": "m", "results": m_runs}, indent=2)
        )
        (out_dir / "param_z_sensitivity.json").write_text(
            json.dumps({"experiment": payload["experiment"], "parameter": "z", "results": z_runs}, indent=2)
        )
        (out_dir / "param_theta_sensitivity.json").write_text(
            json.dumps(
                {"experiment": payload["experiment"], "parameter": "theta", "results": theta_runs},
                indent=2,
            )
        )

    return payload
