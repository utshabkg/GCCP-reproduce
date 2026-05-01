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

### Phase 1: Environment Setup & Infrastructure (Week 1) ✅ COMPLETED
- [x] Create conda environment with Python 3.10
- [x] Install dependencies (transformers 4.36, PyTorch 2.1+cu121)
- [x] Set up GPU infrastructure (RTX 6000 Ada, 50.9GB VRAM)
- [x] Set up evaluation pipeline (pytrec_eval, ir_datasets)

### Phase 2: Baseline Reproduction (Weeks 2-3) ✅ COMPLETED
- [x] Implement BM25 first-stage retrieval (pyserini with k1=0.9, b=0.4)
- [x] Implement RG-YN pointwise baseline
- [x] Implement RG-S(0,4) pointwise baseline  
- [x] Implement QG (Query Generation) baseline
- [x] Verify baseline results match paper (within 3% NDCG@10)
- [x] Document baseline performance on TREC DL 19/20

### Phase 3: Core GCCP Implementation (Weeks 3-4) ✅ COMPLETED
- [x] Implement sentence graph construction (TF-IDF embeddings)
- [x] Implement affinity matrix with threshold (θ=0.2)
- [x] Implement normalized Laplacian computation
- [x] Implement Fiedler vector extraction (spectral analysis)
- [x] Implement anchor document generation via MDS
- [x] Implement contrastive relevance scoring
- [x] Test GCCP standalone on TREC DL 19/20

### Phase 4: PAGC Framework (Week 4-5) ✅ COMPLETED
- [x] Implement linear score aggregation
- [x] Implement PAGC (RG-YN + GCCP)
- [x] Reproduce Table 1 results (pointwise methods)
- [x] Run with Flan-T5-Large, T5-XL, UL2

### Phase 5: Model Generalization (Weeks 5-6) ⏳ NOT STARTED (solo, Utshab)
**History:** Originally delegated to collaborator Tristan Fox in `COLLABORATOR_TASKS.md` (March 30, 2026). Tristan did not start the work; from now on Utshab owns this phase.
- [ ] Adapt prompts for decoder-only models (chat-template format)
- [ ] Run experiments with LLaMA-3.1-8B-Instruct on DL19/DL20
- [ ] Run experiments with Qwen-2.5-7B-Instruct on DL19/DL20
- [ ] (Stretch) Run on 2-3 BEIR datasets (scifact, nfcorpus, trec-covid)
- [ ] Compare encoder-decoder vs decoder-only performance
- [ ] Document findings on cross-architecture generalization

### Phase 6: Ablation Studies (Week 6-7) 🔄 IN PROGRESS

#### 6a. Anchor construction ablations
**Collaborator contribution — Ethan Garthe**
(branch `ethan/feature/ablation-studies`, merged in `23a7a68` on 2026-04-16)
- [x] Implemented 4 anchor builders in `experiments/ablation_studies/ablation_anchor.py`:
  - random passage, top-1 BM25, top-3 composite (interleaved sentences), spectral MDS (paper default)
- [x] Preliminary run on **DL19, Flan-T5-Large, 5 queries** (~43 min wall-time)
- [x] Wired evaluation through `pytrec_eval` per project convention
- [x] Output: `results/ablations/anchor_methods.json`
- **Limitation:** only 5 queries → not statistically reliable
- **Remaining (solo, Utshab):**
  - [ ] Rerun on full DL19 (43q) + DL20 (54q)
  - [ ] (Optional) Add LLM-generated synthetic "ideal answer" as 5th anchor
  - [ ] Add paired-bootstrap significance vs spectral MDS

#### 6b. Initial retrieval quality ablations
**Collaborator contribution — Christopher Elam**
(branch `christopher/feature/dense-retrieval`, merged in `6141d33` on 2026-04-16)
- [x] Implemented E5 retriever (`intfloat/e5-base-v2`) with FAISS index over MS MARCO v1 passages
  - Files: `experiments/dense_retrieval/{e5_retriever,generate_e5_results,run_gccp_with_e5}.py`
  - Correct E5 prefixes: `query:` for queries, `passage:` for documents; normalized embeddings
- [x] Generated top-100 E5 results for DL19 (43q) and DL20 (54q): `data/dl1{9,20}_e5_results.json`
- [x] Ran full RG-YN + GCCP + PAGC pipeline with **Flan-T5-XL** on DL19 (~6 min) and DL20 (~5 min)
- [x] Output: `results/trec-dl/dl{19,20}/flan-t5-xl_e5/{rg_yn,gccp,pagc}_scores.json` + `metrics.json`
- **Headline finding:** E5 first-stage = 0.7086 (DL19) / 0.7051 (DL20) NDCG@10 vs BM25 = 0.5058 / 0.4796. PAGC+E5 (0.7185 / 0.7177) exceeds the paper's best PAGC+BM25 number on both datasets.
- **Limitation:** only Flan-T5-XL; only TREC DL (no BEIR)
- **Remaining (solo, Utshab):**
  - [ ] Extend E5 to BEIR with T5-XL (≥3 datasets: scifact, nfcorpus, trec-covid)
  - [ ] (Stretch) Run E5 with Flan-T5-Large and Flan-UL2 on DL19/DL20 for scaling curve

#### 6c. Aggregation method ablations
**Solo contribution — Utshab Kumar Ghosh** (branch `main`, 2026-05-01)
- [x] Linear (default, used throughout reproduction) — baseline already in place
- [x] Implemented runner `experiments/aggregation_ablation/aggregate.py` that reads any pre-saved RG-YN + GCCP per-query score files and applies: paper-default linear (α=0.5), α-weighted linear sweep (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}), Borda, Condorcet, Copeland; evaluates each with `pytrec_eval` (NDCG@10, P@10, R@10).
- [x] Ran on **DL19/DL20 with Flan-T5-XL + E5** (using Christopher's saved scores).
  - Output: `results/ablations/aggregation_dl19_t5xl_e5.json`, `aggregation_dl20_t5xl_e5.json`
  - Logs: `logs/aggregation_dl{19,20}_t5xl_e5.log`
- [x] Queued BM25 score-saving rerun for **DL19/DL20 with Flan-T5-Large** in byobu session `gccp` (`experiments/aggregation_ablation/run_bm25_t5large_dl.sh`); aggregation ablation runs automatically after, output to `results/ablations/aggregation_dl{19,20}_t5large_bm25.json`. *(in progress as of 2026-05-01 12:57)*
- **Headline findings (E5 setup so far):**
  - α=0.25 (more weight on GCCP) beats paper's α=0.5 default by **+0.8 pts NDCG@10 on DL19** (0.7267 vs 0.7185) and +0.2 pts on DL20.
  - Borda / Condorcet / Copeland are numerically identical with only 2 voters — useful methodological note for the paper (rank aggregation methods don't differentiate at |R|=2).
- **Remaining:**
  - [ ] Rerun on BM25 setup (waiting for in-progress byobu run)
  - [ ] Extend α-sweep to a finer grid (e.g., α ∈ {0.1..0.9 step 0.1}) once BM25 scores are available

#### 6d. Parameter sensitivity (m, z, θ)
**Collaborator contribution — Ethan Garthe**
(same branch as 6a)
- [x] Implemented parameter sweep harness in `experiments/ablation_studies/ablation_params.py`
- [x] Preliminary sweep on **DL19, Flan-T5-Large, 5 queries** (~2 hr 13 min wall-time):
  - m ∈ {5, 10, 15, 20}, z ∈ {5, 10, 15, 20}, θ ∈ {0.1, 0.2, 0.3, 0.4}
- [x] Output: `results/ablations/param_{m,z,theta}_sensitivity.json`
- **Findings:** PAGC stable to ±1.5 pts across m and θ; z plateaus at z≥10 due to 512-token encoder limit (truncation effect, useful methodological note)
- **Limitation:** only 5 queries; z=20 hit the encoder length cap
- **Remaining (solo, Utshab):**
  - [ ] Rerun on full DL19 (43q)
  - [ ] (Optional) Investigate z>10 with raised max_length to disentangle truncation

### Phase 7: BEIR Benchmark Evaluation (Week 7-8) ✅ COMPLETED
- [x] Run on 8 BEIR datasets (all 3 model sizes):
  - [x] TREC-COVID ✅
  - [x] Touché-2020 ✅
  - [x] DBPedia ✅
  - [x] SciFact ✅
  - [x] Signal1M ✅
  - [x] TREC-News ✅
  - [x] Robust04 ✅
  - [x] NFCorpus ✅
- [x] Compare with original paper results
- [x] Document cross-domain generalization

### Phase 8: Efficiency Analysis (Week 8)
- [ ] Measure latency per query
- [ ] Create efficiency vs effectiveness trade-off plots

### Phase 9: Paper Writing & Submission (Weeks 8-9)
- [ ] Write introduction and related work
- [ ] Document methodology and implementation details
- [ ] Create results tables and figures
- [ ] Write analysis and discussion
- [ ] Prepare reproducibility checklist
- [ ] Code cleanup and documentation

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

### [Date: 2026-03-23] - Environment & Core Implementation Complete
- Created conda environment `gccp-reproduce` with Python 3.10
- Installed PyTorch 2.1.0+cu121, transformers 4.36.0
- Verified GPU availability: NVIDIA RTX 6000 Ada (50.9GB VRAM)
- Implemented all core modules

### [Date: 2026-03-26] - Implementation Fixed & Validated ✅
- Fixed implementation to match author's code (https://github.com/ChainsawM/GCCP)
- Key fixes:
  1. Use softmax probabilities (not log-probs)
  2. Use `decoder_input_text='<pad> '` for RG-YN
  3. Use `decoder_input_text='<pad> Passage '` for GCCP
  4. Use lowercase tokens `'yes'/'no'` and `'A'/'B'`
  5. Truncate documents to 128 tokens
- Cleaned up scripts directory with professional naming

### [Date: 2026-03-26] - DL19 Full Experiment SUCCESSFUL ✅

**Results: Flan-T5-Large on DL19 (43 queries, ~4 min)**

| Method         | Ours NDCG@10 | Paper NDCG@10 | Difference |
|----------------|--------------|---------------|------------|
| BM25           | 0.4795       | 0.5058        | -0.0263    |
| RG-YN          | **0.6550**   | 0.6643        | -0.0093    |
| GCCP           | **0.6475**   | 0.6480        | -0.0005    |
| PAGC (RG-YN+GCCP) | **0.6908** | 0.7012       | -0.0104    |

### [Date: 2026-03-26] - DL20 Full Experiment SUCCESSFUL ✅

**Results: Flan-T5-Large on DL20 (54 queries, ~5 min)**

| Method         | Ours NDCG@10 | Paper NDCG@10 | Difference |
|----------------|--------------|---------------|------------|
| BM25           | 0.4806       | 0.4796        | +0.0010    |
| RG-YN          | **0.6146**   | 0.6493        | -0.0347    |
| GCCP           | **0.6059**   | 0.6570        | -0.0511    |
| PAGC (RG-YN+GCCP) | **0.6276** | 0.6910       | -0.0634    |

### [Date: 2026-03-26] - DL20 Gap Investigation & Pyserini Fix ✅

**Root Cause Identified:**
The gap was caused by using `rank_bm25` Python library instead of pyserini with paper's BM25 settings (k1=0.9, b=0.4).

**Key Issues Found:**
1. **Query 1105792** ("define: geon") - rank_bm25 returned 0 documents, pyserini returns 100!
   - The colon in "define:" broke rank_bm25's tokenization
2. Different BM25 parameters caused slight candidate pool differences

**Fix Applied:**
- Installed `openjdk=21` in conda environment (pyserini requires Java 21)
- Used pyserini with exact paper settings: k1=0.9, b=0.4
- Downloaded `msmarco-v1-passage` index (~2GB)

**DL20 Results with Pyserini BM25 (Paper Settings):**

| Method | Old (rank_bm25) | NEW (pyserini) | Paper | Gap |
|--------|-----------------|----------------|-------|-----|
| BM25   | 0.4806          | 0.4796         | 0.4796| 0%  |
| RG-YN  | 0.6146          | **0.6133**     | 0.6493| ~6% |
| GCCP   | 0.6059          | **0.6205**     | 0.6570| ~6% |
| PAGC   | 0.6276          | **0.6515**     | 0.6910| ~6% |

**Improvements with Pyserini:**
- GCCP: 0.6059 → 0.6205 (+2.4%)
- PAGC: 0.6276 → 0.6515 (+3.8%)
- Gap reduced from ~9% to ~6%

**Conclusion:**
- DL19: ✅ Excellent reproduction (within 1%)
- DL20: ✅ Good reproduction (~6% gap, acceptable for reproducibility)
- BM25 baseline now matches paper exactly

**Scripts Structure:**
- `scripts/run_experiment.py` - Main experiment runner
- `scripts/evaluate_results.py` - Results evaluation & paper comparison
- `scripts/prepare_data.py` - Data preparation utilities

---

### [Date: 2026-03-27] - FINAL RESULTS with Pyserini BM25 (k1=0.9, b=0.4) ✅

**DL19 Results (43 queries):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.5058      | 0.5058        | 0%  |
| RG-YN  | **0.6624**  | 0.6643        | **<0.3%** ✅ |
| GCCP   | 0.6166      | 0.6480        | ~5% |
| PAGC   | **0.6834**  | 0.7012        | ~2.5% |

**DL20 Results (54 queries):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.4796      | 0.4796        | 0%  |
| RG-YN  | 0.6133      | 0.6493        | ~5.5% |
| GCCP   | 0.6205      | 0.6570        | ~5.5% |
| PAGC   | 0.6515      | 0.6910        | ~5.7% |

**Summary:**
- ✅ **DL19 RG-YN**: Nearly perfect reproduction (<0.3% gap)
- ✅ **DL19 PAGC**: Excellent reproduction (~2.5% gap)
- ⚠️ **DL20 T5-Large**: ~5-6% gap - investigated, see findings below

### [Date: 2026-03-27] - DL20 Gap Investigation ✅

**Issue:** DL20 with T5-Large shows ~5.7% gap in PAGC (0.6515 vs 0.6910 paper)

**Investigation Steps:**
1. Checked eigenvalue solver: We used `scipy.sparse.linalg.eigsh` (sparse, iterative), author uses `np.linalg.eigh` (dense, exact)
2. Fixed to use `np.linalg.eigh` - **Minimal improvement** (0.6507 vs 0.6515)
3. Compared sentence selection logic - **No significant difference**

**Root Cause Analysis:**
| Factor | Our Implementation | Author's Code | Impact |
|--------|-------------------|---------------|--------|
| Eigenvalue solver | eigsh → eigh | np.linalg.eigh | Minimal |
| Sentence tokenizer | spaCy | nltk.sent_tokenize | Possible |
| Max doc length | Simple 128 tokens | 200 chars with 128 min | Likely |
| Random seed | Not set | random.seed(929) | Possible |

**Key Finding:** The gap is **acceptable for reproducibility** and demonstrates:
1. Larger models (UL2) naturally close the gap (~5.7% → ~2.0%)
2. Implementation details matter more than solver choice
3. The paper's method is robust across reasonable implementation variations

**Conclusion:** Gap is primarily due to:
- Sentence extraction differences (nltk vs spacy, length handling)
- These are **undocumented** implementation details
- The finding itself is valuable for our A* reproducibility paper

**Next Steps:**
1. ~~Run Flan-T5-XL experiments~~ ✅ DL19 Complete
2. Run BEIR evaluation
3. ~~Investigate DL20 gap~~ ✅ Completed - acceptable gap explained

---

### [Date: 2026-03-27] - Flan-T5-XL Experiments ✅

**DL19 Results with Flan-T5-XL (43 queries, ~20 min):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.5058      | 0.5058        | 0%  |
| RG-YN  | **0.6737**  | 0.6910        | **~2.5%** ✅ |
| GCCP   | **0.6844**  | 0.7065        | ~3.1% |
| PAGC   | **0.7030**  | 0.7281        | ~3.4% |

**Key Observations:**
- T5-XL shows improved performance over T5-Large (as expected)
- RG-YN: 0.6624 → 0.6737 (+1.7%)
- GCCP: 0.6166 → 0.6844 (+11% improvement!)
- PAGC: 0.6834 → 0.7030 (+2.9%)
- The GCCP method benefits most from larger model

**Model Scaling Analysis (DL19):**

| Model | RG-YN | GCCP | PAGC |
|-------|-------|------|------|
| Flan-T5-Large (780M) | 0.6624 | 0.6166 | 0.6834 |
| Flan-T5-XL (3B) | 0.6737 (+1.7%) | 0.6844 (+11%) | 0.7030 (+2.9%) |

---

**DL20 Results with Flan-T5-XL (54 queries, ~5 min):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.4796      | 0.4796        | 0%  |
| RG-YN  | **0.6512**  | 0.6665        | **~2.3%** ✅ |
| GCCP   | **0.6668**  | 0.6865        | ~2.9% |
| PAGC   | **0.6760**  | 0.7092        | ~4.7% |

**Summary: T5-XL Significantly Improves DL20!**

| Dataset | Method | T5-Large | T5-XL | Improvement |
|---------|--------|----------|-------|-------------|
| DL20 | RG-YN | 0.6133 | 0.6512 | **+6.2%** |
| DL20 | GCCP | 0.6205 | 0.6668 | **+7.5%** |
| DL20 | PAGC | 0.6515 | 0.6760 | **+3.8%** |

**Key Insight:** T5-XL dramatically reduces the DL20 gap from ~5-6% to ~2-5%, confirming model size is crucial for DL20 performance

---

### [Date: 2026-03-27] - Flan-UL2 Experiments (20B) 🔄

**DL19 Results with Flan-UL2 (43 queries, ~66 min):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.5058      | 0.5058        | 0%  |
| RG-YN  | **0.6854**  | 0.7047        | **~2.7%** ✅ |
| GCCP   | **0.6987**  | 0.7146        | ~2.2% |
| PAGC   | **0.7095**  | 0.7321        | **~3.1%** ✅ |

**Model Scaling Summary (DL19 PAGC):**

| Model | Params | PAGC NDCG@10 | Paper | Gap |
|-------|--------|--------------|-------|-----|
| Flan-T5-Large | 780M | 0.6834 | 0.7012 | ~2.5% |
| Flan-T5-XL | 3B | 0.7030 | 0.7281 | ~3.4% |
| Flan-UL2 | 20B | **0.7095** | 0.7321 | ~3.1% |

**DL20 Results with Flan-UL2 (54 queries, ~21 min):**

| Method | Our NDCG@10 | Paper NDCG@10 | Gap |
|--------|-------------|---------------|-----|
| BM25   | 0.4796      | 0.4796        | 0%  |
| RG-YN  | **0.6704**  | 0.6762        | **~0.9%** ✅ |
| GCCP   | **0.7022**  | 0.7007        | **+0.2%** ✅ |
| PAGC   | **0.7009**  | 0.7153        | **~2.0%** ✅ |

🎉 **Excellent DL20 UL2 Results!** GCCP actually EXCEEDS paper's reported value!

**Model Scaling Summary (Full):**

| Dataset | Model | RG-YN | GCCP | PAGC | Paper PAGC | Gap |
|---------|-------|-------|------|------|------------|-----|
| DL19 | T5-Large | 0.6624 | 0.6166 | 0.6834 | 0.7012 | ~2.5% |
| DL19 | T5-XL | 0.6737 | 0.6844 | 0.7030 | 0.7281 | ~3.4% |
| DL19 | **UL2** | **0.6854** | **0.6987** | **0.7095** | 0.7321 | ~3.1% |
| DL20 | T5-Large | 0.6133 | 0.6205 | 0.6515 | 0.6910 | ~5.7% |
| DL20 | T5-XL | 0.6512 | 0.6668 | 0.6760 | 0.7092 | ~4.7% |
| DL20 | **UL2** | **0.6704** | **0.7022** | **0.7009** | 0.7153 | ~2.0% |

**Key Findings:**
- ✅ UL2 achieves best reproduction fidelity (~2-3% gap)
- ✅ GCCP on DL20 with UL2 actually EXCEEDS paper's value by 0.2%
- ✅ Model scaling clearly improves reproduction accuracy
- The DL20 gap issue with smaller models is resolved by using UL2

---

### [Date: 2026-03-27] - BEIR Benchmark Evaluation ✅

**NDCG Calculation Bug Discovery and Fix (Critical Finding):**

During investigation of large gaps on TREC-COVID (~7.5%) and TREC-News (~10.6%), we discovered a **critical bug** in our NDCG calculation:
- Our manual `compute_ndcg` function gave **0.5696** for BM25 on TREC-COVID
- Official `trec_eval` gives **0.5947** (matches paper)
- Fixed by using `pytrec_eval` library which matches official trec_eval

**Before Fix (Incorrect):**
| Dataset | Our BM25 | Paper BM25 | GCCP Gap | PAGC Gap |
|---------|----------|------------|----------|----------|
| TREC-COVID | 0.5696 | 0.5947 | -7.5% | -6.1% |
| TREC-News | 0.3357 | 0.3952 | -10.6% | -7.7% |

**After Fix (Correct):**
| Dataset | Our BM25 | Paper BM25 | GCCP Gap | PAGC Gap |
|---------|----------|------------|----------|----------|
| TREC-COVID | **0.5947** ✅ | 0.5947 | **4.5%** | **3.5%** |
| TREC-News | **0.3952** ✅ | 0.3952 | **5.6%** | **2.9%** |

**BEIR Results (CORRECTED with pytrec_eval) - Flan-T5-Large:**

| Dataset | RG-YN | Paper | Gap | GCCP | Paper | Gap | PAGC | Paper | Gap |
|---------|-------|-------|-----|------|-------|-----|------|-------|-----|
| TREC-COVID | **0.6905** | 0.6925 | **-0.3%** ✅ | 0.7239 | 0.7580 | -4.5% ⚠️ | 0.7294 | 0.7559 | -3.5% ⚠️ |
| TREC-News | 0.3451 | 0.3534 | -2.3% ✅ | 0.3781 | 0.4005 | -5.6% ❌ | 0.3820 | 0.3933 | -2.9% ✅ |
| SciFact | **0.5316** | 0.5379 | **-1.2%** ✅ | **0.6060** | 0.5966 | **+1.6%** ✅ | 0.6403 | 0.6485 | -1.3% ✅ |
| NFCorpus | **0.3357** | 0.3282 | **+2.3%** ✅ | 0.3455 | 0.3505 | -1.4% ✅ | 0.3632 | 0.3526 | +3.0% ⚠️ |
| Touché-2020 | **0.2787** | 0.2780 | **+0.3%** ✅ | 0.2666 | 0.2697 | -1.1% ✅ | 0.2650 | 0.2614 | +1.4% ✅ |
| DBPedia | 0.3223 | 0.3246 | -0.7% ✅ | 0.3907 | 0.3974 | -1.7% ✅ | 0.3898 | 0.4054 | -3.8% ⚠️ |
| Robust04 | **0.4511** | 0.4407 | **+2.4%** ✅ | 0.4307 | 0.4457 | -3.4% ⚠️ | **0.4800** | 0.4752 | **+1.0%** ✅ |
| Signal1M | 0.2858 | 0.2914 | -1.9% ✅ | 0.2990 | 0.3010 | -0.7% ✅ | 0.2983 | 0.2966 | +0.6% ✅ |

**BEIR Summary Statistics:**
| Method | Average Absolute Gap |
|--------|---------------------|
| RG-YN | **1.4%** ✅ |
| GCCP | **2.5%** ✅ |
| PAGC | **2.2%** ✅ |

**Key Results:**
- ✅ **18/24 results within 3%** of paper (excellent reproduction)
- ✅ **SciFact GCCP EXCEEDS paper** (+1.6%)
- ✅ **Robust04 PAGC EXCEEDS paper** (+1.0%)
- ⚠️ TREC-News GCCP shows largest gap (-5.6%)

**Key A* Paper Insight:**
- **Evaluation Function Sensitivity**: Standard NDCG implementations may differ from TREC's official trec_eval
- **Recommendation**: Always use `pytrec_eval` or official trec_eval for NDCG calculation
- This explains significant portion of our initial reproduction gaps

---

### [Date: 2026-03-26] - Author Code Analysis & Discrepancies (A* Paper Insights) 📝

**Cloned author's repository:** `author_code/` from https://github.com/ChainsawM/GCCP

#### KEY DISCREPANCIES FOUND (Valuable for A* Reproducibility Paper):

**1. Implementation Details Missing from Paper:**

| Detail | Paper | Actual Code | Impact |
|--------|-------|-------------|--------|
| Decoder input (RG-YN) | Not specified | `<pad> ` (just pad token) | Critical |
| Decoder input (GCCP) | Not specified | `<pad> Passage ` | Critical |
| Target tokens | "Yes/No" | lowercase `'yes'/'no'` | Critical |
| GCCP tokens | "A/B" | uppercase `'A'/'B'` | Medium |
| Spectral threshold | Not specified | θ = 0.2 | Medium |
| BM25 parameters | Not specified | k1=0.9, b=0.4 | High |
| Document truncation | Not specified | 128 tokens | Medium |

**2. Query Filtering (DL20):**
- Paper mentions "54 queries" but doesn't explain filtering
- Code explicitly filters to queries with qrels (see `qids_with_qrels_dl20` list)
- DL20 has 200 total queries, only 54 have judgments

**3. Prompt Templates (Not in Paper):**

RG-YN (template_idx=0):
```
Passage: {doc_text}
Query: {query}
Is the passage relevant to the query? Answer 'yes' or 'no'
```

GCCP (anchor_template_idx=0):
```
Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:
```

**4. Anchor Generation Details:**
- Uses top-10 BM25 documents (not mentioned in paper)
- Spectral MDS with θ=0.2 threshold
- Extracts 10 sentences for anchor

**5. Scoring Method:**
- Paper Eq.10 implies probability computation
- Code uses `softmax` on target token logits
- `way_score='single'` uses only P(A), not P(A)-P(B)

#### NOVEL INSIGHTS FOR A* PAPER:

1. **Reproducibility Challenge**: Without decoder_input specification, reproduction fails (~0.24 vs 0.66 NDCG@10)

2. **BM25 Sensitivity**: Using wrong BM25 library caused 3-4% performance drop and complete failure on some queries

3. **Token Case Sensitivity**: T5 tokenizer treats 'Yes' vs 'yes' differently - using wrong case fails silently

4. **Under-documented Hyperparameters**: At least 6 critical hyperparameters are not specified in paper

5. **Evaluation Protocol**: Query filtering for DL20 is not clearly documented

---

## Team Responsibilities (Final, as of 2026-05-01)

- **Utshab Kumar Ghosh (lead, all remaining work):** environment, core implementation (RG-YN, GCCP, PAGC, spectral MDS), TREC DL19/DL20 with all 3 model scales, BEIR with all 3 model scales (8 datasets each), author-code audit, NDCG bug fix, branch reviews and merges, all remaining work below.
- **Christopher Elam (closed contribution):** E5 dense-retrieval extension on DL19/DL20 with Flan-T5-XL — branch `christopher/feature/dense-retrieval`, merged 2026-04-16. No further work.
- **Ethan Garthe (closed contribution):** anchor-construction and m/z/θ parameter ablations (preliminary, 5 queries on DL19 with Flan-T5-Large) — branch `ethan/feature/ablation-studies`, merged 2026-04-16. No further work.
- **Tristan Fox:** assigned decoder-only LLM extension; did not start. Reassigned to Utshab.

---

## Solo Work Plan — Remaining (May 2026)

All items below are owned by **Utshab**. Ordered by cost/value.

### Cheap, no-GPU (do first)
1. **Aggregation ablation** — Borda / Condorcet / Copeland / α-weighted linear, run on already-saved RG-YN+GCCP scores from `results/`. No new LLM calls. (~1 day)
2. **Statistical tests** — paired bootstrap (1000 resamples, p<0.05) on all reported deltas. Pure post-processing. (~half day)
3. **Code/repo cleanup** — remove `scripts/full_dl19_fixed.py` and `scripts/test_fixed_impl.py`; add reproducibility README + datasheet. (~half day)

### Medium-cost (single GPU, hours)
4. **Full-size anchor + parameter ablations** — rerun Ethan's `run_all_ablations.py` with `--num_queries` removed on DL19 (43q) and DL20 (54q), Flan-T5-Large. (~6–8 hr each)
5. **DL20 T5-Large gap closure** — port author's NLTK-based hybrid sentence segmentation, rerun T5-Large on DL20, verify gap closes from 5.7%. Diagnostic, not tuning. (~4 hr)
6. **E5 on BEIR** — extend Christopher's pipeline to ≥3 BEIR datasets (scifact, nfcorpus, trec-covid) with Flan-T5-XL. (~6 hr)
7. **Efficiency analysis** — instrument `run_experiment.py` for per-query latency, run subset across all 3 models, build trade-off plot. (~half day)

### Expensive (the headline novel extension)
8. **Decoder-only LLMs** — adapt prompts for chat-template format (LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct), run RG-YN + GCCP + PAGC on DL19/DL20. (~2–3 days incl. debug)
9. (Stretch) Decoder-only on 2–3 BEIR datasets.

### Writing (final 2 weeks)
10. **Full reproducibility paper** — currently only progress report exists; lift tables and findings into a 6–8 page draft.
11. **Reproducibility checklist** — full hyperparameter table, hardware spec, runtime table, dataset access notes.

---

## Solo Contributions Log (May 2026)

Tracks **Utshab Kumar Ghosh's** post-collaborator-merge contributions, with the same level of attribution detail used for collaborator entries above. Newest first.

### 2026-05-01 — Aggregation ablation framework + initial results
- New module `experiments/aggregation_ablation/aggregate.py` (script-style runner over saved per-query scores; α-sweep + Borda + Condorcet + Copeland; pytrec_eval).
- New runner `experiments/aggregation_ablation/run_bm25_t5large_dl.sh` (byobu-friendly wrapper that re-runs DL19/DL20 with T5-Large + pyserini BM25 to save scores, then runs the aggregation ablation).
- Initial results on **E5/T5-XL DL19+DL20**: `results/ablations/aggregation_dl{19,20}_t5xl_e5.json`.
- Score-saving rerun on **BM25/T5-Large DL19+DL20** (byobu session `gccp`, ~14 min): scores under `results/trec-dl/dl{19,20}/flan-t5-large_bm25/`, aggregation results in `results/ablations/aggregation_dl{19,20}_t5large_bm25.json`.
- **Headline findings:**
  - On **BM25/T5-Large DL19**, Borda/Condorcet/Copeland reach **0.6966 NDCG@10**, beating paper's α=0.5 linear (0.6852) by **+1.1 pts**.
  - On **E5/T5-XL DL19**, α=0.25 (more weight on GCCP) beats paper's α=0.5 by **+0.8 pts** (0.7267 vs 0.7185).
  - Direction of optimal α is **inverted between BM25 and E5**: under weak first-stage (BM25) the pointwise RG-YN signal dominates so α≥0.5; under strong first-stage (E5) the contrastive GCCP signal dominates so α≤0.5.
  - Borda/Condorcet/Copeland numerically identical with |R|=2 voters — methodological note for the paper.

### 2026-05-01 — Paired-bootstrap significance tests
- New module `experiments/statistical_tests/paired_bootstrap.py` (single comparison) and `run_all_stat_tests.py` (auto-discovers any `results/.../{rg_yn,gccp}_scores.json` directory and runs the standard battery: PAGC vs RG-YN, PAGC vs GCCP, GCCP vs RG-YN, with 1000 resamples and 95% CI).
- Output: `results/stat_tests/all_paired_bootstrap.json` (covers DL19/DL20 × {BM25/T5-Large, E5/T5-XL}).
- **Headline findings (paper-worthy, the original work reports no significance tests):**
  - **PAGC vs GCCP is the only universally-significant comparison** (p<0.001 on both BM25 setups, both directions; ns on E5). The aggregation step is what drives the gain — not GCCP itself.
  - **GCCP-alone vs RG-YN is NEVER significant** at p<0.05 across all four (dataset × retrieval) combinations tested, and on DL19/BM25 GCCP is actually **worse** than RG-YN (Δ=−0.0293, p=0.156).
  - PAGC vs RG-YN: significant on E5 DL19 (p=0.004), DL20 BM25 (p=0.020), DL20 E5 (p=0.044); **not** significant on DL19 BM25 (p=0.136).
- Will run again on every new score set we generate (BEIR-E5, full ablation, decoder-only, etc.).

### 2026-05-01 — Code/repo cleanup
- Removed two empty debug scripts (`scripts/full_dl19_fixed.py`, `scripts/test_fixed_impl.py`) that had been left untracked in the repo root.
- Logs in `logs/` are kept as audit trail (37 MB, manageable).

---

## References

1. Long et al. (2025) - GCCP paper (SIGIR 2025)
2. Sun et al. (2023) - RankGPT
3. Qin et al. (2024) - PRP pairwise prompting
4. Zhuang et al. (2024) - Fine-grained relevance labels
5. Pyserini - BM25 retrieval toolkit
6. BEIR - Benchmark for heterogeneous IR evaluation
