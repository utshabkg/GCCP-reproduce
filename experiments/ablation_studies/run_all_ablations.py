#!/usr/bin/env python3
#Ablation Studies for GCCP

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ablation_studies.ablation_anchor import (
    DEFAULTS,
    compute_rg_yn_results,
    load_dataset,
    run_anchor_method_ablation,
)

from experiments.ablation_studies.ablation_params import run_parameter_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GCCP ablation studies")
    parser.add_argument("--dataset", default="dl19", choices=["dl19"], help="Dataset to use")
    parser.add_argument(
        "--model",
        default="flan-t5-large",
        choices=["flan-t5-large", "flan-t5-xl", "flan-ul2"],
        help="Model to use",
    )
    parser.add_argument("--num_queries", type=int, default=None, help="Optional limit for quick testing")
    parser.add_argument(
        "--output_dir",
        default=str(Path("results") / "ablations"),
        help="Directory to write ablation outputs",
    )
    parser.add_argument("--seed", type=int, default=929, help="Seed for random anchor selection")
    args = parser.parse_args()


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries, _, bm25_results = load_dataset(args.dataset)
    qids = list(queries.keys())[: args.num_queries] if args.num_queries else list(queries.keys())
    rg_yn_results = compute_rg_yn_results(
        queries,
        bm25_results,
        qids,
        model_name=args.model,
        max_doc_length=DEFAULTS["max_doc_length"],
    )

    anchor_results = run_anchor_method_ablation(
        dataset=args.dataset,
        model_name=args.model,
        num_queries=args.num_queries,
        output_path=output_dir / "anchor_methods.json",
        seed=args.seed,
        rg_yn_results=rg_yn_results,
    )

    param_results = run_parameter_sensitivity(
        dataset=args.dataset,
        model_name=args.model,
        num_queries=args.num_queries,
        output_dir=output_dir,
        rg_yn_results=rg_yn_results,
    )

    # out files
    summary = {
        "anchor_methods_file": str(output_dir / "anchor_methods.json"),
        "param_m_file": str(output_dir / "param_m_sensitivity.json"),
        "param_z_file": str(output_dir / "param_z_sensitivity.json"),
        "param_theta_file": str(output_dir / "param_theta_sensitivity.json"),
        "num_queries": anchor_results["experiment"]["num_queries"],
        "model": args.model,
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
