# GCCP Reproducibility Study

Reproducing "Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information" (SIGIR 2025)

## Overview

This project reproduces the GCCP (Global-Context Comparative Pointwise) method for zero-shot document ranking. We validate the original results and document critical implementation details that were not specified in the paper.

## Key Results

- **TREC-DL Average Gap:** 3.0% (10/18 results within 3%)
- **BEIR T5-Large Average Gap:** 2.1%
- **BEIR T5-XL Average Gap:** 4.2%
- **BEIR UL2 Average Gap:** 3.2%

See `results/FINAL_SUMMARY.md` for detailed results.

## Critical Implementation Details Discovered

| Detail | Paper | Actual Code | Impact |
|--------|-------|-------------|--------|
| Decoder input (RG-YN) | Not specified | `'<pad> '` | Critical |
| Decoder input (GCCP) | Not specified | `'<pad> Passage '` | Critical |
| Target tokens | "Yes/No" | lowercase `'yes'/'no'` | Critical |
| GCCP tokens | "A/B" | uppercase `'A'/'B'` | Medium |
| Spectral threshold | Not specified | θ = 0.2 | Medium |
| BM25 parameters | Not specified | k1=0.9, b=0.4 | Medium |

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate gccp-reproduce

# Or manual setup
conda create -n gccp-reproduce python=3.10
conda activate gccp-reproduce
pip install -r requirements.txt
```

## Running Experiments

### TREC-DL
```bash
python scripts/run_experiment.py --dataset dl19 --model flan-t5-large
python scripts/run_experiment.py --dataset dl20 --model flan-t5-xl
```

### BEIR
```bash
python scripts/run_beir.py --dataset scifact --model flan-t5-large
python scripts/run_beir.py --dataset all --model flan-t5-xl
```

## Repository Structure

```
├── src/
│   ├── pointwise/rankers.py    # RG-YN implementation
│   ├── gccp/gccp_ranker.py     # GCCP implementation
│   └── gccp/spectral_mds.py    # Anchor generation
├── scripts/
│   ├── run_experiment.py       # TREC-DL experiments
│   └── run_beir.py             # BEIR experiments
├── results/
│   ├── trec-dl/                # TREC-DL results
│   ├── beir/                   # BEIR results
│   └── FINAL_SUMMARY.md        # Summary of all results
├── data/                       # Queries, qrels, BM25 runs
├── PLAN.md                     # Detailed project plan
└── COLLABORATOR_TASKS.md       # Task distribution
```

## Models Tested

- Flan-T5-Large (780M)
- Flan-T5-XL (3B)
- Flan-UL2 (20B)

## Datasets

**TREC-DL:** DL19 (43 queries), DL20 (54 queries)

**BEIR:** SciFact, NFCorpus, TREC-COVID, TREC-News, Touché-2020, DBPedia, Robust04, Signal1M

## Citation

Original paper:
```
@inproceedings{gccp2025,
  title={Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information},
  author={...},
  booktitle={SIGIR},
  year={2025}
}
```

## License

MIT
