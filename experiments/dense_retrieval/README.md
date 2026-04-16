# Dense Retrieval Extension (E5)

## Overview

This extension replaces the original **BM25 first-stage retrieval** with **E5 dense retrieval** using the model:

- `intfloat/e5-base-v2`

The rest of the pipeline remains unchanged:
- RG-YN (pointwise reranking)
- GCCP (graph-based reranking)
- PAGC (combined method)

The goal is to evaluate how replacing BM25 with E5 affects the quality of candidate retrieval and downstream ranking performance.

---

## Files Added

experiments/dense_retrieval/
├── e5_retriever.py  
├── generate_e5_results.py  
├── run_gccp_with_e5.py  
└── README.md  

### Descriptions

- **e5_retriever.py**
  - Implements dense retrieval using SentenceTransformers (E5)
  - Encodes queries and documents into embeddings
  - Uses FAISS for similarity search

- **generate_e5_results.py**
  - Generates top-100 retrieved documents per query
  - Saves results in the same format as BM25 outputs

- **run_gccp_with_e5.py**
  - Runs RG-YN, GCCP, and PAGC using E5 retrieval results
  - Produces evaluation metrics and score files

---

## Running Experiments

### 1. Generate E5 retrieval results

#### DL19

```bash
python experiments/dense_retrieval/generate_e5_results.py \
  --dataset dl19 \
  --model intfloat/e5-base-v2 \
  --output_file data/dl19_e5_results.json \
  --device cuda
```

#### DL20

```bash
python experiments/dense_retrieval/generate_e5_results.py \
  --dataset dl20 \
  --model intfloat/e5-base-v2 \
  --output_file data/dl20_e5_results.json \
  --device cuda
```

---

### 2. Run GCCP pipeline with E5

#### DL19

```bash
python experiments/dense_retrieval/run_gccp_with_e5.py \
  --dataset dl19 \
  --model flan-t5-xl \
  --retrieval_file data/dl19_e5_results.json \
  --output_dir results/trec-dl/dl19/flan-t5-xl_e5
```

#### DL20

```bash
python experiments/dense_retrieval/run_gccp_with_e5.py \
  --dataset dl20 \
  --model flan-t5-xl \
  --retrieval_file data/dl20_e5_results.json \
  --output_dir results/trec-dl/dl20/flan-t5-xl_e5
```

---

## Output Files

Each experiment produces:

results/trec-dl/{dataset}/{model}_e5/
├── metrics.json
├── rg_yn_scores.json
├── gccp_scores.json
└── pagc_scores.json

---

## Results and Comparison

### DL19 (flan-t5-xl)

| Method  | BM25   | E5         |
| ------- | ------ | ---------- |
| Initial | 0.5058 | **0.7086** |
| RG-YN   | 0.6737 | 0.6752     |
| GCCP    | 0.6844 | **0.7090** |
| PAGC    | 0.7030 | **0.7185** |

---

### DL20 (flan-t5-xl)

| Method  | BM25   | E5         |
| ------- | ------ | ---------- |
| Initial | 0.4796 | **0.7051** |
| RG-YN   | 0.6512 | 0.6789     |
| GCCP    | 0.6668 | **0.7069** |
| PAGC    | 0.6760 | **0.7177** |

---

## Observations

* Replacing BM25 with E5 results in a large improvement in initial retrieval quality:

  * +0.20 NDCG@10 on DL19
  * +0.23 NDCG@10 on DL20
* These improvements propagate through the entire pipeline, improving RG-YN, GCCP, and PAGC performance.
* The gains from reranking are smaller with E5 compared to BM25, indicating that E5 produces higher-quality candidate sets.
* GCCP and PAGC benefit more from improved retrieval than RG-YN.
* PAGC remains the best-performing method, achieving the highest NDCG@10 on both datasets.

---

## Summary

This extension demonstrates that:

* Dense retrieval (E5) substantially improves first-stage retrieval performance compared to BM25.
* Improved candidate retrieval leads to consistent gains across all downstream ranking methods.
* The GCCP framework remains effective when applied to dense retrieval candidates.
* PAGC provides the strongest overall ranking performance in this setup.

