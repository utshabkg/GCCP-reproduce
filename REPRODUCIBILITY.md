# Reproducibility Checklist

Companion document for the reproducibility paper *"Reproducing GCCP: A
Forensic Study of Zero-Shot Pointwise Ranking with Post-Aggregated
Global Context"*. Targets the standard ACL/SIGIR reproducibility
checklist.

**Repository:** https://github.com/utshabkg/GCCP-reproduce

---

## 1. Environment

### 1a. Hardware
- **GPUs:** 2× NVIDIA RTX 6000 Ada Generation (48 GB VRAM each).
- **CPU/RAM:** sufficient for IO-bound steps; not the bottleneck.
- **Storage:** ~150 GB used across model caches, BEIR Lucene indices,
  and result files.

### 1b. Software
Two conda environments are required because the encoder-decoder pipeline
uses an older `transformers` (4.36) for legacy T5 stability and the
decoder-only extension needs `transformers ≥ 4.43` for LLaMA-3 / Qwen-2.5
tokenizer support:

| Env             | Python | torch       | transformers | Used by                      |
|-----------------|--------|-------------|--------------|------------------------------|
| `gccp-reproduce`| 3.10   | 2.1.0+cu121 | 4.36.0       | All Flan-T5 / Flan-UL2 work  |
| `gccp-decoder`  | 3.10   | 2.4.0+cu121 | 4.49.0       | LLaMA-3 / Qwen-2.5 / Mistral |

Reproduce both:

```bash
conda env create -f environment.yml          # gccp-reproduce
conda create -n gccp-decoder python=3.10 -y
conda run -n gccp-decoder pip install \
    torch==2.4.0 'transformers>=4.46,<4.50' accelerate sentencepiece \
    protobuf pytrec_eval scikit-learn nltk numpy tqdm
```

### 1c. Java
Pyserini's anserini JARs require **Java 21**. The conda env install
already pins `openjdk=21`. If you hit `pyjnius` errors, double-check
`echo $JAVA_HOME`.

---

## 2. Data

All datasets are publicly available and accessed via Pyserini's prebuilt
indices and `ir_datasets`. No private data.

| Dataset       | Source            | Queries | Notes                                |
|---------------|-------------------|---------|--------------------------------------|
| TREC DL 2019  | MS MARCO v1       | 43      | `pyserini msmarco-v1-passage` index  |
| TREC DL 2020  | MS MARCO v1       | 54      | same index                           |
| BEIR (8 sets) | BEIR              | varies  | per-dataset prebuilt Lucene indices  |

**Exact BM25 parameters:** `k1 = 0.9`, `b = 0.4`. (Not specified in
original paper; recovered from author's code.)

**Pre-computed top-100 BM25 results checked into the repo** for
DL19/DL20: `data/dl{19,20}_pyserini_bm25.json`. BEIR retrievals are
generated on the fly inside `scripts/run_beir.py`.

---

## 3. Models

All models loaded via HuggingFace `transformers`. FP16 on GPU.

| Model                              | Size  | License        | Cached at                                                                  |
|------------------------------------|-------|----------------|----------------------------------------------------------------------------|
| google/flan-t5-large               | 780M  | apache-2.0     | `/media/4TB/share/models/huggingface/`                                    |
| google/flan-t5-xl                  | 3B    | apache-2.0     | same                                                                       |
| google/flan-ul2                    | 20B   | apache-2.0     | same                                                                       |
| meta-llama/Meta-Llama-3.1-8B-Instruct | 8B | llama-3.1      | local snapshot at `/media/20TB/shared/models/meta-llama/Llama-3.1-8B-Instruct/` |
| Qwen/Qwen2.5-7B-Instruct           | 7B    | qwen-license   | local snapshot at `/media/20TB/shared/models/qwen/Qwen2.5-7B-Instruct/`   |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B    | apache-2.0     | local snapshot at `/media/20TB/shared/models/mistralai/Mistral-7B-Instruct-v0.3/` |
| ~~Qwen/Qwen2.5-72B-Instruct-AWQ~~  | 72B   | qwen-license   | local; **dropped from final results** -- see note below                   |

LLaMA-3.1-8B-Instruct is gated on Hugging Face but a snapshot was
already present on local university storage; we ran it from the local
snapshot under `TRANSFORMERS_OFFLINE=1` rather than re-downloading.
All four 7--8B decoder-only runs (LLaMA, Qwen, Mistral, $\times$ DL19/DL20)
were obtained from these local snapshots.

\textbf{Qwen-2.5-72B-AWQ stretch goal.} We attempted the AWQ-quantized
72B as a scaling data point. Loading the model requires the
\texttt{autoawq} package, whose 0.2.7+ versions force a PyTorch
upgrade to $\geq 2.5.1$ that breaks the rest of the gccp-decoder
environment (CUDA build incompatibility). Older autoawq versions
need \texttt{transformers.models.qwen3} which is unavailable in our
pinned transformers 4.49. We backed the change out and report 72B as
future work.

---

## 4. Hyperparameters

All settings are the paper's defaults plus the seven undocumented
details we recovered (Section 5 of the paper).

| Knob                        | Value                                | Note              |
|-----------------------------|--------------------------------------|-------------------|
| BM25 candidates per query   | 100                                  |                   |
| BM25 $k_1, b$               | 0.9, 0.4                             | undoc.            |
| Anchor pool $m$             | 10                                   |                   |
| Anchor sentences $z$        | 10                                   |                   |
| Spectral threshold $\theta$ | 0.2                                  | undoc.            |
| Document truncation         | 128 tokens                           | undoc.            |
| Decoder input (RG-YN, T5)   | `'<pad> '`                           | undoc.            |
| Decoder input (GCCP, T5)    | `'<pad> Passage '`                   | undoc.            |
| Target tokens (RG-YN)       | lowercase `'yes', 'no'`              | undoc.            |
| Target tokens (GCCP)        | uppercase `'A', 'B'`                 | undoc.            |
| Sentence segmentation       | NLTK `sent_tokenize`                 | undoc.            |
| Per-doc length cap          | 200 chars OR first 128, hybrid       | undoc.            |
| Random seed                 | 929                                  | undoc.            |

For decoder-only models we additionally:
- Use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- Prime GCCP response with the literal string `'Passage '` (analogous to
  T5 decoder-input).
- Aggregate probability mass over case/space variants of yes/no/A/B.

---

## 5. Compute Budget

| Component                                      | Wall-clock (approx.) |
|-------------------------------------------------|----------------------|
| Full TREC DL 19/20 with Flan-T5-Large          | ~14 min              |
| Full TREC DL 19/20 with Flan-T5-XL             | ~25 min each         |
| Full TREC DL 19/20 with Flan-UL2               | ~70 min each         |
| Full BEIR (8 sets) with Flan-T5-Large          | ~6 hrs               |
| Full BEIR (8 sets) with Flan-T5-XL             | ~12 hrs              |
| Full BEIR (8 sets) with Flan-UL2               | ~5 days (no batching)|
| Anchor + parameter ablations (DL19+DL20)       | ~1.5 hrs             |
| Decoder-only DL19+DL20 (Qwen + Mistral)        | ~2 hrs               |
| Efficiency measurement (10 queries × 3 models) | ~30 min              |

Inference is **sequential** (no batching) in both the original code
and ours. Batching would speed up at least 2–4× and is logged as
future work.

---

## 6. Evaluation

- **Primary metric:** NDCG@10 via `pytrec_eval`. Matches official
  `trec_eval` to ≤ 4 decimal places. **Do not use hand-rolled NDCG**
  (Section 5 of the paper for why).
- **Secondary:** P@10, Recall@10, NDCG@100.
- **Statistical tests:** paired bootstrap, 1000 resamples, seed 929,
  two-sided p-values, 95% CI on the mean delta. Implementation in
  `experiments/statistical_tests/paired_bootstrap.py`.

---

## 7. Reproducing Each Result

| Result                          | Command                                                                                                  |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| TREC-DL T5-Large                | `bash experiments/aggregation_ablation/run_bm25_t5large_dl.sh`                                           |
| BEIR T5-{Large/XL/UL2}          | `bash scripts/run_all_beir.sh` (per-model invocation in script)                                          |
| Aggregation ablation            | `python experiments/aggregation_ablation/aggregate.py --rg_yn_scores ... --gccp_scores ... --qrels ...` |
| Stat tests                      | `python experiments/statistical_tests/run_all_stat_tests.py`                                             |
| Full anchor + param ablation    | `bash experiments/ablation_studies/run_full_ablations.sh`                                                |
| DL20 author-MDS                 | `python experiments/dl20_gap_closure/run_dl20_author_exact.py`                                           |
| Decoder-only LLMs               | `bash experiments/decoder_only_models/run_all_decoder.sh`                                                |
| Efficiency                      | `bash experiments/efficiency_analysis/run_all_efficiency.sh`                                             |

Each command writes to `results/...` and `logs/...` deterministically;
expected file layout is documented in [PLAN.md](PLAN.md).

---

## 8. Negative / Partial Results (kept on the record)

- **DL20 / Flan-T5-Large gap (5.7% PAGC) is not fully closed** even
  with a faithful port of the author's NLTK + 200/128 hybrid sentence
  segmentation. Our best effort closes ~0.7 pts; the rest is
  unexplained. Not hidden from the paper -- we treat it as a finding.

- **LLaMA-3.1-8B-Instruct** could not be downloaded without HF auth.
  Replaced with Mistral-7B-Instruct-v0.3.

- **Spectral MDS does not beat top-1 BM25 anchor** on full DL19/DL20
  with Flan-T5-Large under our reproduction. The paper claims it
  does (Table 5). We have not yet been able to reconcile the
  discrepancy and we report both numbers honestly.

---

## 9. Variance

All deterministic given seed 929 except for HuggingFace model
downloads (which sometimes resolve different shard files mid-flight).
Score files are checked into the repo under `results/`, so any
downstream analysis is bit-exact.
