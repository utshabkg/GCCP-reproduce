# When Do Anchor-Based Pointwise LLM Rerankers Help?

Code and data for an anonymous submission under double-blind review.

We study GCCP/PAGC ([Long et al., SIGIR 2025](https://doi.org/10.1145/3726302.3730061)) as a representative anchor-based pointwise reranker and identify the boundary conditions under which the method actually helps. The work is reproduction-first: we reimplement the pipeline from the paper text alone, audit the gap against the released code, then run controlled extensions along five research questions.

## Findings

**RQ1 — Reproduction and operational sensitivity.** The pipeline reproduces within 1.6% mean absolute nDCG@10 on TREC-DL and 1.9–4.5% (per-scale mean) on 8 BEIR datasets. **Eight operational choices not stated in the original paper together explain the full 0.66 → 0.24 nDCG@10 collapse** that a paper-only reimplementation produces on DL19/Flan-T5-Large; three are make-or-break (each silently broken), five are performance-relevant. The single largest (target-token case) recovers 0.24 → 0.55 in isolation.

**RQ2 — Statistical claims under correction.** Replacing the original per-cell t-tests with paired bootstrap (10,000 resamples, seed 929) and Holm-Bonferroni correction across 22 primary settings:
- PAGC vs RG-YN: Holm-significant in **12/22** (all positive).
- GCCP vs RG-YN: directional in **19/22** (sign-test p ≪ 0.001), per-cell Holm-significant in **3/22**.
- PAGC vs GCCP-alone: Holm-significant in **5/22** with mixed signs, including a Holm-significant **negative** on DBPedia-Entity (Δ = −0.0144, p_Holm = 0.032).

**RQ3 — Retriever quality moderates reranker value.** Under BM25 aggregation is consistently useful; under E5 dense retrieval, PAGC is statistically tied with GCCP-alone on 7/8 BEIR sets. The DBPedia-Entity negative **replicates under a second dense retriever** (BGE-base-en-v1.5; k=3 confirmatory Holm family) with larger magnitude (Δ = −0.0190, p_Holm < 0.001). Per-cell Kendall τ between RG-YN and GCCP scores is +0.04 higher on E5 cells than BM25 cells — the agreement-based account partially explains the moderation but does not single out DBPedia.

**RQ4 — Anchor construction.** Spectral multi-document summarization does not justify its complexity: spectral MDS wins on **0 of 8 BEIR datasets** and finishes last on 3. A top-3 sentence-interleaved composite is simpler, faster, and stronger in aggregate.

**RQ5 — Backbone transfer.** The mechanism transfers to decoder-only LLMs (LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct-v0.3) and to a 4-bit AWQ-quantized 72B model on a single 48 GB GPU. The DBPedia-Entity negative survives encoder-decoder → decoder-only, 3B → 72B parameters, and FP16 → 4-bit AWQ — three independent axes of replication.

## Eight undocumented implementation choices

### Make-or-break (pipeline runs but produces broken rankings)

| Choice | Default in paper | Required value |
|---|---|---|
| T5 decoder input string | unspecified | `<pad> ` (RG-YN), `<pad> Passage ` (GCCP) |
| Target-token case (RG-YN) | "Yes/No" | lowercase `yes`/`no` |
| Per-query min-max normalization before linear aggregation | not in Eq. 11 | required for components to share scale |

### Performance-relevant (do not collapse the pipeline but materially shift results)

| Choice | Default in paper | Required value |
|---|---|---|
| Target-token case (GCCP) | "A/B" | uppercase `A`/`B` |
| Spectral threshold θ | unspecified | 0.2 |
| BM25 parameters | unspecified | k1 = 0.9, b = 0.4 |
| Document truncation | unspecified | 128 tokens |
| nDCG implementation | unspecified | `pytrec_eval` (a hand-rolled routine differs by ~2.5 pts on TREC-COVID) |

## Setup

Two conda environments are used. The encoder-decoder pipeline uses an older `transformers` for Flan-T5 stability; the decoder-only extension needs a newer one for LLaMA-3 / Qwen-2.5 tokenizer support.

```bash
# Encoder-decoder pipeline (Flan-T5-Large/XL, Flan-UL2, E5/BGE retrieval)
conda env create -f environment.yml
conda activate gccp-reproduce

# Decoder-only extension (LLaMA-3.1, Qwen-2.5, Mistral) — separate env
conda create -n gccp-decoder python=3.10 -y
conda activate gccp-decoder
pip install torch==2.4.0 'transformers>=4.46,<4.50' accelerate sentencepiece \
            protobuf pytrec_eval scikit-learn nltk numpy tqdm \
            sentence-transformers faiss-cpu ir_datasets
```

Java 21 is required for Pyserini's anserini JARs (`environment.yml` pins `openjdk=21`).

## Models and datasets

**Encoder-decoder rerankers (`gccp-reproduce` env):** Flan-T5-Large (780M), Flan-T5-XL (3B), Flan-UL2 (20B).

**Decoder-only rerankers (`gccp-decoder` env):** LLaMA-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Qwen-2.5-72B-Instruct-AWQ (4-bit AWQ on a single 48 GB GPU).

**TREC-DL:** DL19 (43 queries), DL20 (54 queries) over MS MARCO v1 passages.

**BEIR (8 paper-aligned subsets):** SciFact, NFCorpus, TREC-COVID, TREC-News, Touché-2020, DBPedia-Entity, Robust04, Signal-1M.

**First-stage retrievers:** BM25 (Pyserini, k1 = 0.9, b = 0.4), E5-base-v2, BGE-base-en-v1.5.

## Reproducing key results

```bash
# TREC-DL reproduction (RQ1)
python scripts/run_experiment.py --dataset dl19 --model flan-t5-xl
python scripts/run_experiment.py --dataset dl20 --model flan-t5-xl

# BEIR with E5 retrieval (RQ3 primary)
python experiments/beir_e5/generate_beir_e5.py --dataset dbpedia-entity
python experiments/beir_e5/rerank_beir_e5.py    --dataset dbpedia-entity --model flan-t5-xl

# BEIR with BGE retrieval (RQ3 robustness check)
python experiments/beir_e5/generate_beir_bge.py --dataset dbpedia-entity
python experiments/beir_e5/rerank_beir_bge.py   --dataset dbpedia-entity --model flan-t5-xl

# Paired bootstrap + Holm correction across all settings (RQ2)
python experiments/statistical_tests/run_all_stat_tests.py

# Anchor and parameter ablations (RQ4)
bash experiments/ablation_studies/run_full_ablations.sh

# Decoder-only LLMs (RQ5)
bash experiments/decoder_only_models/run_all_decoder.sh
```

Per-query nDCG@10 scores for every cell are released under `results/beir/<dataset>/<model>_<retriever>/`. Bootstrap p-values and Holm-corrected significance are in `results/stat_tests/all_paired_bootstrap.json`. The BGE confirmatory family is at `results/stat_tests/bge_robustness_k3_holm.json` and per-cell Kendall τ at `results/stat_tests/rgyn_gccp_kendall_tau.json`.

## Repository layout

```
src/                                 Core method implementations
├── pointwise/rankers.py             RG-YN pointwise scoring
├── gccp/gccp_ranker.py              Contrastive A/B scoring
├── gccp/spectral_mds.py             Spectral anchor builder
└── pagc/aggregation.py              Linear / Borda aggregation

experiments/                         Per-RQ experiment scripts
├── beir_e5/                         BEIR with E5 and BGE retrieval (RQ3)
├── decoder_only_models/             LLaMA, Qwen, Mistral, Qwen-72B-AWQ (RQ5)
├── statistical_tests/               Paired bootstrap + Holm (RQ2)
├── ablation_studies/                Anchor + parameter sweeps (RQ4)
├── aggregation_ablation/            α-sweep and Borda comparison
├── efficiency_analysis/             Wall-clock latency
├── pagc_qsg/                        3-component PAGC-RS-YN-GCCP variant
├── listwise_baseline/               llm-rankers RankGPT re-run attempt
├── dl20_gap_closure/                Author-MDS sentence-segmentation port
├── trec_dl/                         TREC-DL pipeline drivers
└── beir/                            BEIR pipeline drivers

scripts/                             Top-level experiment drivers
├── run_experiment.py                TREC-DL driver
├── run_beir.py                      BEIR driver
└── run_all_beir.sh                  Full 8-set BEIR sweep

results/
├── beir/<dataset>/<model>_<retriever>/   {rg_yn,gccp,pagc}_scores.json + metrics.json
├── trec-dl/dl{19,20}/<model>_<retriever>/   same layout
├── stat_tests/                      Bootstrap, Holm, Kendall τ summaries
├── figures/                         Figures corresponding to the paper
└── ablations/                       Per-anchor and per-hyperparameter sweeps

data/                                First-stage retrieval candidates, queries, qrels
```

## Citation

This repository accompanies an anonymous double-blind submission. The original method being studied is:

```bibtex
@inproceedings{long2025precise,
  title     = {Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information},
  author    = {Long, Kehan and Li, Shasha and Xu, Chen and Tang, Jintao and Wang, Ting},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year      = {2025},
  doi       = {10.1145/3726302.3730061}
}
```

## License

MIT
