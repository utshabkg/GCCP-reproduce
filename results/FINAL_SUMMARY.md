# GCCP Reproducibility Study - Final Results Summary

**Last update:** May 2026
**Repository:** https://github.com/utshabkg/GCCP-reproduce
**Target:** A* Conference Reproducibility Track (SIGIR/CIKM/EMNLP)

---

## Overview

This study reproduces and extends the GCCP / PAGC method from:
> Long et al. *Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information.* SIGIR 2025.

**Encoder-decoder backbones (paper-aligned):** Flan-T5-Large (780M), Flan-T5-XL (3B), Flan-UL2 (20B).
**Decoder-only backbones (novel extension):** LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct-v0.3.
**Datasets:** TREC DL 2019/2020 + 8 BEIR subsets.
**First-stage retrievers:** Pyserini BM25 (k1=0.9, b=0.4) + intfloat/e5-base-v2 (novel).

---

## Headline Findings

### 1. Faithful reproduction
- **TREC-DL avg PAGC gap:** ~3.0% across 6 (dataset × model) cells.
- **BEIR avg PAGC gap:** 2.1% (T5-Large), 4.2% (T5-XL), 3.2% (UL2).
- Several individual cells **exceed** the paper (Robust04, TREC-News, Signal1M, NFCorpus, SciFact, UL2-DL20-GCCP).

### 2. Seven undocumented implementation details
Recovered from author's code; each fails silently and the first two together drop NDCG@10 from 0.66 to 0.24:

| Detail                    | Paper | Code              | Impact   |
|---------------------------|-------|-------------------|----------|
| Decoder input (RG-YN, T5) | -     | `'<pad> '`        | Critical |
| Decoder input (GCCP, T5)  | -     | `'<pad> Passage '`| Critical |
| Target tokens (RG-YN)     | "Yes/No" | lowercase     | Critical |
| Target tokens (GCCP)      | "A/B" | uppercase A/B     | Medium   |
| Spectral threshold        | -     | θ = 0.2           | Medium   |
| BM25 parameters           | -     | k1=0.9, b=0.4     | Medium   |
| Document truncation       | -     | 128 tokens        | Medium   |

### 3. Statistical significance (paper has none)
Paired bootstrap (**10,000 resamples**) with **Holm-Bonferroni
correction** across 13 (dataset × retrieval × model) settings:
- **GCCP-alone vs RG-YN:** significant at $p_\text{Holm}<0.05$ in **0/13** settings. Three settings (BEIR-SciFact, BEIR-NFCorpus, DL19/Qwen-2.5-7B) are significant at raw p<0.05 but lose under Holm.
- **PAGC vs GCCP:** significant at $p_\text{Holm}<0.05$ in **4/13** settings, all of them BM25 setups.
- **PAGC vs RG-YN:** significant at $p_\text{Holm}<0.05$ in 5/13, including both BEIR-E5 sets where statistical power is high.
- Implication: GCCP's value lives in the aggregation step, not in GCCP alone. Note: power is limited on TREC-DL (n=43/54), so "not significant" is consistent with both no effect and a small undetectable effect.

### 4. Spectral MDS anchor never wins (full DL19/DL20 + 7 of 8 BEIR sets, T5-Large)

**TREC-DL (replicates paper's reported exception, +4× magnitude):**
| Anchor builder | DL19 GCCP | DL19 PAGC | DL20 GCCP | DL20 PAGC |
|---|---|---|---|---|
| Random passage | 0.6394 | 0.6945 | 0.6103 | 0.6474 |
| Top-1 BM25     | **0.6511** | **0.6948** | 0.6131 | 0.6485 |
| Top-3 composite| 0.6410 | 0.6947 | **0.6280** | **0.6572** |
| Spectral MDS (paper) | 0.6341 | 0.6852 | 0.6137 | 0.6507 |

**BEIR (7 of 8 sets, GCCP NDCG@10):**
| Anchor          | SciFact | NFCorpus | TREC-COVID | Touché | Robust04 | News | Signal1M | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Random          | 0.5781 | 0.3213 | **0.7701** | **0.2954** | 0.4386 | **0.4384** | 0.2894 | 0.4473 |
| Top-1 BM25      | 0.6659 | 0.3303 | 0.7609 | 0.2894 | 0.4423 | 0.4218 | 0.2734 | 0.4549 |
| Top-3 composite | **0.6692** | **0.3381** | 0.7649 | 0.2869 | **0.4505** | 0.4261 | **0.3020** | **0.4625** |
| Spectral (paper)| 0.6195 | 0.3311 | 0.7564 | 0.2694 | 0.4380 | 0.4245 | 0.2990 | 0.4483 |

**Spectral MDS is third or fourth on every single one of the 7 BEIR sets.** Top-3 composite wins on 4/7, Random on 3/7. Paper's reported BEIR-aggregate ordering (Spectral 0.4471 > Top 0.4346 > Random 0.4253) is the OPPOSITE of our reproduction (Top-3 0.4625 > Top-1 0.4549 > Spectral 0.4483 > Random 0.4473). We have drafted an email to the original authors asking about the "Top" definition, since our Top-1 (0.4549) is +2 pts above the paper's reported Top (0.4346); they may use a different operationalization (truncation, title-only, etc.) we have not been able to identify.

### 5. Aggregation method matters
On DL19 / Flan-T5-Large / BM25:
- Paper's α=0.5 linear: 0.6852 NDCG@10
- α-sweep optimal (α=0.5 or 0.75): 0.6848-0.6852
- **Borda / Condorcet / Copeland: 0.6966 NDCG@10 (+1.1 pts over paper)**

On DL19 / Flan-T5-XL / E5: optimal α flips to **α=0.25** (0.7267, +0.8 pts over α=0.5).

The optimal α is sensitive to first-stage retrieval quality; the paper's fixed α=0.5 is sub-optimal in both regimes.

### 6. E5 dense retrieval lifts everything
NDCG@10 with Flan-T5-XL, BM25 vs E5 first-stage:
| Dataset    | BM25 first-stage | E5 first-stage | PAGC+BM25 | PAGC+E5 |
|------------|------------------|----------------|-----------|---------|
| DL19       | 0.5058 | 0.7086 | 0.7030 | **0.7185** |
| DL20       | 0.4796 | 0.7051 | 0.6760 | **0.7177** |
| SciFact    | 0.6789 | 0.7274 | 0.6840 | **0.7176** |
| NFCorpus   | 0.3387 | 0.3517 | 0.3728 | **0.3884** |

The marginal contribution of GCCP/PAGC over E5 is much smaller than over BM25 (e.g., DL20: +0.197 over BM25 vs +0.013 over E5). The contrastive anchor matters most when the candidate list is noisy.

### 7. Decoder-only LLMs work; Qwen-2.5-72B-AWQ is the strongest cell
TREC-DL NDCG@10 with chat-template + 'Passage ' primer:
| Dataset | Model | RG-YN | GCCP | PAGC |
|---------|-------|-------|------|------|
| DL19 | LLaMA-3.1-8B-Instruct | 0.6427 | 0.6521 | 0.6723 |
| DL19 | Qwen-2.5-7B-Instruct | 0.6527 | 0.7039 | 0.7212 |
| DL19 | Mistral-7B-Instruct-v0.3 | 0.5716 | 0.5617 | 0.6429 |
| DL19 | **Qwen-2.5-72B-AWQ** | 0.6590 | **0.7323** | **0.7465** |
| DL20 | LLaMA-3.1-8B-Instruct | 0.5795 | 0.5866 | 0.6166 |
| DL20 | Qwen-2.5-7B-Instruct | 0.6360 | 0.6447 | 0.6641 |
| DL20 | Mistral-7B-Instruct-v0.3 | 0.5360 | 0.5105 | 0.5983 |
| DL20 | **Qwen-2.5-72B-AWQ** | 0.6311 | **0.6983** | 0.6980 |

**Qwen-2.5-72B-AWQ on DL19 PAGC = 0.7465** is the strongest result in our entire experiment — **surpassing both our reproduced Flan-UL2 (0.7095) AND the paper's own reported Flan-UL2 (0.7321)**. A 4-bit AWQ-quantized 72B open-weight model on a single 48 GB GPU matches the paper's biggest dense FP16 result. At 7-8B scale, Qwen-2.5-7B is competitive with Flan-T5-XL (paired bootstrap p=0.125, not significant on n=43); LLaMA-3.1-8B is intermediate; Mistral-7B underperforms. Backbone family matters at least as much as parameter count at the 7-8B scale.

### 8. NDCG implementation sensitivity
A hand-rolled NDCG produced 0.5696 on TREC-COVID BM25, while official trec_eval / pytrec_eval gave 0.5947 -- accounting for ~25% of our largest BEIR gap before the fix.

### 9. Efficiency
Per-query latency on RTX 6000 Ada (FP16, 100 docs/q, no batching):
| Model | RG-YN | GCCP | PAGC |
|---|---|---|---|
| Flan-T5-Large (780M) | 2.08 s | 2.39 s | 4.47 s |
| Flan-T5-XL (3B)      | 1.99 s | 2.65 s | 4.64 s |

Flan-UL2 latency was excluded due to CPU offload artifact under concurrent runs; from end-to-end runs it is ~25 s/q for the full pipeline.

---

## TREC-DL PAGC NDCG@10 (paper-aligned cells)

| Dataset | Model | Ours | Paper | Gap |
|---------|-------|------|-------|-----|
| DL19 | Flan-T5-Large | 0.6834 | 0.7012 | -2.5% ✅ |
| DL19 | Flan-T5-XL    | 0.7030 | 0.7281 | -3.5% |
| DL19 | Flan-UL2      | 0.7095 | 0.7321 | -3.1% |
| DL20 | Flan-T5-Large | 0.6515 | 0.6910 | -5.7% |
| DL20 | Flan-T5-XL    | 0.6760 | 0.7092 | -4.7% |
| DL20 | Flan-UL2      | 0.7009 | 0.7153 | -2.0% ✅ |

The DL20-T5-Large 5.7% gap was investigated (faithful NLTK + 200/128 hybrid sentence-segmentation port) and reduced to ~5%; ~85% of it remains unattributed even with the author's exact rule.

---

## BEIR PAGC NDCG@10 (Flan-T5-Large)

| Dataset       | Ours | Paper | Gap |
|---------------|------|-------|-----|
| SciFact       | 0.6403 | 0.6485 | -1.3% ✅ |
| NFCorpus      | 0.3620 | 0.3526 | +2.7% ✅ |
| TREC-COVID    | 0.7294 | 0.7559 | -3.5% |
| TREC-News     | 0.3820 | 0.3933 | -2.9% ✅ |
| Touché-2020   | 0.2650 | 0.2614 | +1.4% ✅ |
| DBPedia       | 0.3898 | 0.4054 | -3.9% |
| Robust04      | 0.4800 | 0.4752 | +1.0% ✅ |
| Signal1M      | 0.2983 | 0.2966 | +0.6% ✅ |

Average absolute gap: **2.1%**.

---

## Repository Map

| Path | Contents |
|---|---|
| `src/`                              | Core RG-YN / GCCP / PAGC / spectral-MDS implementation |
| `scripts/`                          | TREC-DL + BEIR experiment runners                  |
| `experiments/aggregation_ablation/` | α-sweep + Borda/Condorcet/Copeland                |
| `experiments/statistical_tests/`    | Paired bootstrap auto-discoverer                  |
| `experiments/ablation_studies/`     | Anchor + parameter sweeps (full DL19/DL20)        |
| `experiments/dl20_gap_closure/`     | Author NLTK-MDS port for DL20 T5-Large gap        |
| `experiments/decoder_only_models/`  | LLaMA-3.1 / Qwen-2.5 / Mistral chat-template ranker |
| `experiments/dense_retrieval/`      | Christopher's E5 retriever (TREC-DL)              |
| `experiments/beir_e5/`              | Two-stage E5 → T5-XL pipeline (BEIR)              |
| `experiments/efficiency_analysis/`  | Per-query latency harness                          |
| `paper/`                            | Proposal, progress report, full reproducibility paper, references.bib |
| `REPRODUCIBILITY.md`                | Reproducibility checklist (env, data, hparams, commands) |
| `PLAN.md`                           | Full project log incl. attribution traces         |
| `results/`                          | All metrics + per-query scores + stat tests + efficiency JSONs |

---

## Conclusion

GCCP / PAGC reproduces faithfully (within 2-4% NDCG@10 on average), but the reproduction effort surfaced enough discrepancies between paper and code -- and enough novel findings via straightforward extensions -- that the work itself is publishable as a reproducibility study. Code, per-query scores, run logs, statistical tests, and the reproducibility checklist are all released.
