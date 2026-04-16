# Ablation Studies

This folder contains collaborator 2's ablation experiments for anchor
construction and GCCP hyperparameter sensitivity.

# Scope

They reuse the existing GCCP, RG-YN, and PAGC implementations while varying:

- anchor construction method
- m: number of top documents used to build the anchor
- z: number of anchor sentences
- theta: sentence-similarity threshold for spectral MDS

# Files

- ablation_anchor.py: anchor-construction ablation helpers and runner
- ablation_params.py: parameter sensitivity sweeps
- run_all_ablations.py: main entry point for all ablations

# Methods

- random_document: randomly choose one document from the top-10 BM25 pool
- top1_bm25: use the top-ranked BM25 document as the anchor
- top3_composite: interleave sentences from the top 3 BM25 documents to form a text-only composite anchor
- spectral_mds: default GCCP spectral anchor


Note: implementation operationalizes average as sentence level anchor built from the top 3 docs.


# Output

1. results/ablations/anchor_methods.json
2. results/ablations/param_m_sensitivity.json
3. results/ablations/param_z_sensitivity.json
4. results/ablations/param_theta_sensitivity.json

## Flow

- Parses CLI args at run_all_ablations.py
- Creates output folder
- Loads dataset and BM25 candidates
- Computes RG-YN scores for selected queries
- Runs anchor ablation 
- Runs parameter sweeps
- Writes summary JSON for analysis

## Usage

From the repository root:

```bash
python experiments/ablation_studies/run_all_ablations.py --dataset dl19 --model flan-t5-large
```


## Evaluation

All evaluation uses `pytrec_eval`, matching the project's documented
recommendation for official NDCG-compatible metrics.

## Future Implementations / Fixes

- Upgrade maximum sequence length beyond 512 for z sensitivity @ 20 


