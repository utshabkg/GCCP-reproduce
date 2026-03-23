# GCCP Reproducibility Study - Project Plan

## A* Conference Reproducibility Track Target

**Target Venue:** ACL/EMNLP/NeurIPS Reproducibility Track (or SIGIR Reproducibility Track if available)  
**Base Paper:** "Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information" (SIGIR 2025)

---

## Executive Summary

This project reproduces and extends the GCCP (Global-Consistent Comparative Pointwise Ranking) method for zero-shot document ranking. The original paper proposes using an anchor document constructed via spectral-based multi-document summarization as a global reference point for pointwise LLM ranking.

### Key Contributions of Original Paper:
1. **GCCP**: Introduces anchor document for global-consistent comparative pointwise ranking
2. **PAGC**: Post-Aggregation framework combining GCCP with traditional pointwise scores
3. Achieves competitive performance with comparative methods at pointwise efficiency

### Our Reproducibility Goals:
1. **Reproduce** original results on TREC DL 2019/2020 and BEIR benchmarks
2. **Extend** to new LLM families (LLaMA-3, Qwen-2.5) not tested in original paper
3. **Ablation** studies on anchor construction and aggregation strategies
4. **Novel Extension**: Test with dense retrieval (E5) instead of BM25

---

## Project Timeline (Weeks 1-9)

### Phase 1: Environment Setup & Infrastructure (Week 1)
- [ ] Create conda environment with Python 3.10
- [ ] Install dependencies (transformers, pyserini, scipy, etc.)
- [ ] Set up GPU infrastructure (Ada A6000 GPUs)
- [ ] Clone reference repository for verification
- [ ] Download TREC DL and BEIR datasets
- [ ] Set up evaluation pipeline (pytrec_eval)

### Phase 2: Baseline Reproduction (Weeks 2-3)
- [ ] Implement BM25 first-stage retrieval with Pyserini
- [ ] Implement RG-YN pointwise baseline
- [ ] Implement RG-S(0,4) pointwise baseline  
- [ ] Implement QG (Query Generation) baseline
- [ ] Verify baseline results match paper (within 1-2% NDCG@10)
- [ ] Document baseline performance on TREC DL 19/20

### Phase 3: Core GCCP Implementation (Weeks 3-4)
- [ ] Implement sentence graph construction (TF-IDF embeddings)
- [ ] Implement affinity matrix with threshold
- [ ] Implement normalized Laplacian computation
- [ ] Implement Fiedler vector extraction (spectral analysis)
- [ ] Implement anchor document generation via MDS
- [ ] Implement contrastive relevance scoring
- [ ] Test GCCP standalone on TREC DL 19/20

### Phase 4: PAGC Framework (Week 4-5)
- [ ] Implement linear score aggregation
- [ ] Implement PAGC-QYG (QG + RG-YN + GCCP)
- [ ] Implement PAGC-QSG (QG + RG-S + GCCP)
- [ ] Reproduce Table 1 results (pointwise methods)
- [ ] Reproduce Table 2 results (vs comparative methods)
- [ ] Statistical significance testing (paired t-test p≤0.05)

### Phase 5: Model Generalization (Weeks 5-6)
- [ ] Adapt prompts for decoder-only models
- [ ] Run experiments with LLaMA-3.1-8B-Instruct
- [ ] Run experiments with Qwen-2.5-7B-Instruct
- [ ] Run experiments with Phi-3-mini (3.8B)
- [ ] Compare encoder-decoder vs decoder-only performance
- [ ] Document findings on cross-architecture generalization

### Phase 6: Ablation Studies (Week 6-7)
- [ ] Anchor construction ablations:
  - [ ] Random document as anchor
  - [ ] Top-1 BM25 document as anchor
  - [ ] LLM-generated synthetic "ideal answer" as anchor
  - [ ] Our spectral MDS approach
- [ ] Initial retrieval quality ablations:
  - [ ] BM25 retrieval (baseline)
  - [ ] E5-base-v2 dense retrieval (novel extension)
- [ ] Aggregation method ablations:
  - [ ] Linear (default)
  - [ ] Borda, Condorcet, Copeland
  - [ ] Non-uniform weighting exploration
- [ ] Parameter sensitivity (m, z values)

### Phase 7: BEIR Benchmark Evaluation (Week 7-8)
- [ ] Run on 8 BEIR datasets:
  - [ ] Covid
  - [ ] Touche
  - [ ] DBPedia
  - [ ] SciFact
  - [ ] Signal
  - [ ] News
  - [ ] Robust04
  - [ ] NFCorpus
- [ ] Compare with original paper results
- [ ] Document cross-domain generalization

### Phase 8: Efficiency Analysis (Week 8)
- [ ] Measure latency per query
- [ ] Estimate API cost (GPT-4 pricing model)
- [ ] Create efficiency vs effectiveness trade-off plots
- [ ] Compare with comparative methods (RankGPT, PRP)
- [ ] Document computational requirements

### Phase 9: Paper Writing & Submission (Weeks 8-9)
- [ ] Write introduction and related work
- [ ] Document methodology and implementation details
- [ ] Create results tables and figures
- [ ] Write analysis and discussion
- [ ] Prepare reproducibility checklist
- [ ] Code cleanup and documentation
- [ ] Create reproducibility package

---

## Technical Implementation Details

### Models to Use:
1. **Encoder-Decoder (Original)**:
   - Flan-T5-Large (780M params) - ~1.5GB GPU
   - Flan-T5-XL (3B params) - ~6GB GPU FP16
   - Flan-UL2 (20B params) - ~40GB GPU FP16

2. **Decoder-Only (Extension)**:
   - LLaMA-3.1-8B-Instruct - ~16GB GPU FP16
   - Qwen-2.5-7B-Instruct - ~14GB GPU FP16
   - Phi-3-mini (3.8B) - ~8GB GPU FP16

### Key Hyperparameters:
- `m = 10`: Top-m documents for MDS
- `z = 10`: Number of sentences in anchor
- `θ`: Affinity threshold for sentence graph
- BM25 retrieves top-100 candidates per query

### Evaluation Metrics:
- **Primary**: NDCG@10
- **Secondary**: Precision@10, Recall@10, NDCG@100
- **Efficiency**: Latency (sec/query), Cost ($/query)

### Hardware Resources:
- 2× NVIDIA Ada A6000 (48GB each)
- Can run Flan-UL2 with model parallelism across both GPUs
- Decoder-only 7-8B models fit comfortably on single GPU

---

## Directory Structure

```
GCCP-reproduce/
├── PLAN.md                    # This file
├── data/
│   ├── trec_dl19/            # TREC DL 2019 queries/qrels
│   ├── trec_dl20/            # TREC DL 2020 queries/qrels
│   └── beir/                 # BEIR benchmark datasets
├── src/
│   ├── retrieval/
│   │   ├── bm25_retrieval.py
│   │   └── dense_retrieval.py
│   ├── pointwise/
│   │   ├── rg_yn.py          # RG-YN baseline
│   │   ├── rg_s.py           # RG-S(0,4) baseline
│   │   └── qg.py             # Query Generation
│   ├── gccp/
│   │   ├── mds.py            # Multi-document summarization
│   │   ├── spectral.py       # Spectral analysis
│   │   ├── anchor.py         # Anchor document generation
│   │   └── gccp_ranker.py    # GCCP implementation
│   ├── pagc/
│   │   └── aggregation.py    # Score aggregation methods
│   └── evaluation/
│       └── metrics.py        # Evaluation utilities
├── experiments/
│   ├── trec_dl/
│   └── beir/
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/                 # Analysis notebooks
└── paper/                     # Paper draft
```

---

## Key Equations to Implement

### 1. RG-YN Score (Eq. 3)
```
f_RG-YN(q, d_i) = exp(S_Y) / (exp(S_Y) + exp(S_N))
where S_Y = LLM(Yes|q, d_i, P_RG-YN)
```

### 2. Affinity Matrix (Eq. 5)
```
a_{i,j} = cos(e_i, e_j) if cos(e_i, e_j) >= θ else 0
```

### 3. Normalized Laplacian (Eq. 6)
```
L = I - D^{-1/2} A D^{-1/2}
```

### 4. Fiedler Vector (Eq. 7-8)
```
Lv_2 = λ_2 v_2
v_2 = argmin v^T L v (subject to constraints)
```

### 5. Contrastive Score (Eq. 10)
```
f_c(q, d_i, d_a) = LLM(d_i | q, d_i, d_a, P_GCCP)
```

### 6. PAGC Final Score (Eq. 11)
```
f_final(q, d_i) = 1/(|R|+1) * (Σ_R f(q, d_i) + f_c(q, d_i, d_a))
```

---

## Reproducibility Requirements for A* Conference

1. **Complete Code Release**: All implementation with clear documentation
2. **Data Availability**: Scripts to download/prepare all datasets
3. **Exact Hyperparameters**: All settings clearly documented
4. **Hardware Specification**: GPU types, memory, compute time
5. **Statistical Significance**: p-values for all comparisons
6. **Variance Reporting**: Multiple runs with confidence intervals
7. **Negative Results**: Document what didn't work and why
8. **Original vs Reproduced**: Clear comparison tables

---

## Important Notes on Code Reuse

**Question:** Do we need to implement from scratch for acceptance?

**Answer:** For reproducibility tracks, the goal is typically to:
1. **Verify** the original claims independently
2. **Document** any discrepancies found
3. **Extend** the work with new experiments

It is **acceptable** to:
- Use the author's code as **reference** for verification
- Implement your **own version** to ensure understanding
- Clearly cite any code reuse

It is **recommended** to:
- Implement core components independently when feasible
- Cross-validate with author's code for debugging
- Document any differences in implementation

For maximum impact, we will:
- Implement core algorithms from the paper description
- Use author's code for verification/debugging only
- Clearly document our implementation choices

---

## Progress Log

### [Date: 2026-03-23] - Project Initiated
- Reviewed base paper (SIGIR 2025)
- Reviewed project proposal
- Created initial plan
- Next: Environment setup

---

## Team Responsibilities (TBD)

- **Member A**: Environment setup, BM25 pipeline, TREC DL baselines
- **Member B**: GCCP/PAGC core implementation and verification
- **Member C**: Decoder-only model experiments, prompt adaptation
- **Member D**: Ablations, BEIR evaluation, figures/tables
- **All**: Final report and presentation

---

## References

1. Long et al. (2025) - GCCP paper (SIGIR 2025)
2. Sun et al. (2023) - RankGPT
3. Qin et al. (2024) - PRP pairwise prompting
4. Zhuang et al. (2024) - Fine-grained relevance labels
5. Pyserini - BM25 retrieval toolkit
6. BEIR - Benchmark for heterogeneous IR evaluation
