# GCCP Reproducibility Study - Collaborator Task Distribution

**Date:** March 30, 2026  
**Repository:** https://github.com/utshabkg/GCCP-reproduce  
**Target:** A* Conference Reproducibility Track (CIKM/EMNLP/NeurIPS)

---

## 📊 Current Progress Summary

### ✅ Completed Work (by Utshab)

| Task | Status | Details |
|------|--------|---------|
| Environment Setup | ✅ Done | conda env `gccp-reproduce`, PyTorch 2.1+cu121 |
| Core Implementation | ✅ Done | RG-YN, GCCP, PAGC, Spectral MDS |
| Critical Bug Fixes | ✅ Done | 6+ undocumented implementation details discovered |
| TREC DL19/DL20 | ✅ Done | All 3 models (T5-Large, T5-XL, UL2) |
| BEIR with T5-Large model | ✅ Done | All 8 datasets |
| BEIR with T5-XL model | ✅ Done | All 8 datasets |
| BEIR UL2 | 🔄 Running | ~5 days remaining (do not interrupt!) |

### 📈 Results Achieved

**TREC-DL (Gap vs Paper):**
- DL19 PAGC: 2.5-3.1% gap ✅
- DL20 PAGC: 2.0-5.7% gap ✅
- UL2's GCCP on DL20 **exceeded** paper by 0.2%!

**BEIR T5-Large (Avg Absolute Gap):**
- RG-YN: 1.4% ✅
- GCCP: 2.5% ✅  
- PAGC: 2.2% ✅

### 🔑 Key Findings (Valuable for Paper)

1. **6+ Undocumented Implementation Details** discovered from author's code:
   - Decoder input for RG-YN: `'<pad> '`
   - Decoder input for GCCP: `'<pad> Passage '`
   - Target tokens: lowercase `'yes'/'no'`, uppercase `'A'/'B'`
   - Spectral MDS threshold θ = 0.2
   - BM25 parameters: k1=0.9, b=0.4

2. **NDCG Calculation Sensitivity**: Must use `pytrec_eval` (matches official trec_eval)

3. **Model Scaling Effect**: Larger models significantly improve reproduction accuracy

---

## 🎯 Tasks for Collaborators

**⚠️ IMPORTANT RULES:**
- Create your own branch with naming: `yourname/feature/workname`
- Create your own code folder: `experiments/your_work_name/`
- DO NOT modify existing scripts in `src/` or `scripts/`
- You can import from existing modules, but write your experiment code separately
- Mill does NOT have datasets pre-downloaded - you must download them yourself

---

### Collaborator 1: Decoder-Only Models (LLaMA-3, Qwen-2.5)

**Branch:** `alice/feature/decoder-only-models` (replace `alice` with your name)

**Task:** Extend GCCP to decoder-only LLMs (novel extension not in original paper)

**Setup:**
```bash
# Clone and setup
git clone https://github.com/utshabkg/GCCP-reproduce.git
cd GCCP-reproduce
git checkout -b yourname/feature/decoder-only-models

# Setup environment
conda env create -f environment.yml
conda activate gccp-reproduce

# Download datasets on Mill (required!)
# TREC DL19/DL20 via ir_datasets:
python -c "import ir_datasets; ir_datasets.load('msmarco-passage/trec-dl-2019')"
python -c "import ir_datasets; ir_datasets.load('msmarco-passage/trec-dl-2020')"

# Download models (Mill doesn't have them):
# - meta-llama/Meta-Llama-3.1-8B-Instruct
# - Qwen/Qwen2.5-7B-Instruct
```

**Your Work Directory:**
```
experiments/decoder_only_models/
├── decoder_ranker.py      # Your decoder-only implementation
├── run_decoder_dl19.py    # Script to run DL19
├── run_decoder_dl20.py    # Script to run DL20
├── run_decoder_beir.py    # Script for BEIR (optional)
└── README.md              # Notes on your implementation
```

**What to do:**
1. Study existing `src/pointwise/rankers.py` to understand the approach
2. Write your own decoder-only ranker in `experiments/decoder_only_models/`
3. Adapt prompts for chat template format (LLaMA/Qwen use chat templates)
4. Run on DL19 and DL20 first (smaller, faster validation)
5. If successful, run on 2-3 BEIR datasets (scifact, trec-covid, nfcorpus)

**Expected output:**
- `results/trec-dl/dl19/llama3-8b_metrics.json`
- `results/trec-dl/dl20/llama3-8b_metrics.json`
- `results/trec-dl/dl19/qwen2.5-7b_metrics.json`
- `results/trec-dl/dl20/qwen2.5-7b_metrics.json`

---

### Collaborator 2: Ablation Studies

**Branch:** `bob/feature/ablation-studies` (replace `bob` with your name)

**Task:** Run ablation experiments on anchor document construction

**Setup:**
```bash
git clone https://github.com/utshabkg/GCCP-reproduce.git
cd GCCP-reproduce
git checkout -b yourname/feature/ablation-studies

conda env create -f environment.yml
conda activate gccp-reproduce

# Download datasets on Mill (required!)
python -c "import ir_datasets; ir_datasets.load('msmarco-passage/trec-dl-2019')"
```

**Your Work Directory:**
```
experiments/ablation_studies/
├── ablation_anchor.py       # Different anchor construction methods
├── ablation_params.py       # Parameter sensitivity experiments
├── run_all_ablations.py     # Main runner script
└── README.md                # Notes and findings
```

**Ablations to run (on DL19 with T5-Large):**

1. **Anchor Construction Methods:**
   - Random document as anchor (baseline)
   - Top-1 BM25 document as anchor
   - Average of top-3 BM25 documents as anchor
   - Spectral MDS anchor (current default)

2. **Parameter Sensitivity:**
   - Vary m (top docs for anchor): [5, 10, 15, 20]
   - Vary z (sentences in anchor): [5, 10, 15, 20]
   - Vary θ (similarity threshold): [0.1, 0.2, 0.3, 0.4]

**Expected output:**
- `results/ablations/anchor_methods.json`
- `results/ablations/param_m_sensitivity.json`
- `results/ablations/param_z_sensitivity.json`
- `results/ablations/param_theta_sensitivity.json`

---

### Collaborator 3: Dense Retrieval Extension

**Branch:** `carol/feature/dense-retrieval` (replace `carol` with your name)

**Task:** Replace BM25 first-stage with E5 dense retrieval (novel extension)

**Setup:**
```bash
git clone https://github.com/utshabkg/GCCP-reproduce.git
cd GCCP-reproduce
git checkout -b yourname/feature/dense-retrieval

conda env create -f environment.yml
conda activate gccp-reproduce
pip install sentence-transformers faiss-gpu

# Download datasets on Mill (required!)
python -c "import ir_datasets; ir_datasets.load('msmarco-passage/trec-dl-2019')"
python -c "import ir_datasets; ir_datasets.load('msmarco-passage/trec-dl-2020')"

# Download E5 model:
# - intfloat/e5-base-v2 or intfloat/e5-large-v2
```

**Your Work Directory:**
```
experiments/dense_retrieval/
├── e5_retriever.py          # E5 dense retrieval implementation
├── generate_e5_results.py   # Generate top-100 for DL19/DL20
├── run_gccp_with_e5.py      # Run GCCP with E5 results
└── README.md                # Notes and comparison analysis
```

**What to do:**
1. Implement E5 retrieval to get top-100 documents per query
2. Save results in same format as BM25 results (see `data/dl19_pyserini_bm25.json`)
3. Run GCCP pipeline using your E5 retrieval results
4. Compare performance: E5 vs BM25 as first-stage retrieval

**Expected output:**
- `data/dl19_e5_results.json` (E5 retrieval results)
- `data/dl20_e5_results.json`
- `results/trec-dl/dl19/flan-t5-large_e5_metrics.json`
- `results/trec-dl/dl20/flan-t5-large_e5_metrics.json`

---

## 📁 Repository Structure

```
GCCP-reproduce/
├── src/                        # Core implementation (DO NOT MODIFY)
│   ├── pointwise/rankers.py    # RG-YN implementation
│   ├── gccp/gccp_ranker.py     # GCCP implementation
│   └── gccp/spectral_mds.py    # Anchor generation
├── scripts/                    # Main experiment scripts (DO NOT MODIFY)
│   ├── run_experiment.py       # DL19/DL20 experiments
│   └── run_beir.py             # BEIR experiments
├── experiments/                # YOUR CODE GOES HERE
│   ├── decoder_only_models/    # Collaborator 1
│   ├── ablation_studies/       # Collaborator 2
│   └── dense_retrieval/        # Collaborator 3
├── results/
│   ├── trec-dl/{dl19,dl20}/    # TREC-DL results
│   ├── beir/{dataset}/         # BEIR results
│   └── ablations/              # Ablation results
├── data/                       # Queries, qrels, BM25 runs
└── PLAN.md                     # Detailed project plan & findings
```

---

## 🖥️ Mill Setup Instructions (Seeing Documentation would be better as I am not sure of this one below)

```bash
# Request GPU node
srun --partition=gpu --gres=gpu:1 --mem=64G --time=24:00:00 --pty bash

# Or submit batch job
sbatch your_experiment.slurm

# Set HuggingFace cache to avoid quota issues:
export HF_HOME=$HOME/.cache/huggingface
# Or use scratch space if available:
export HF_HOME=/scratch/$USER/huggingface

# Download MS MARCO corpus (needed for document contents):
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')"
```

---

## 📝 Reporting Results

1. **Commit results to your branch regularly**
```bash
git add experiments/your_work/ results/
git commit -m "Add [experiment_name] results"
git push origin yourname/feature/workname
```

2. **Create Pull Request when done** with:
   - Summary of what was run
   - Key results table (markdown)
   - Any issues encountered

3. **Update your README.md** with findings

---

## ⚠️ Critical Notes

- **DO NOT** push to `main` branch directly - use Pull Requests
- **DO NOT** modify files in `src/` or `scripts/` - write your own code
- **DO NOT** interrupt the running UL2 experiment
- **USE** `pytrec_eval` for NDCG calculation (critical for correct results!)
- **READ** `PLAN.md` carefully - it contains important implementation details

---

## 📞 Questions?

Contact Utshab on Discord/Email for:
- Implementation details
- Clarification on findings
- Code review

**Deadline:** Initial results by end of Week 2 (April 12, 2026)
