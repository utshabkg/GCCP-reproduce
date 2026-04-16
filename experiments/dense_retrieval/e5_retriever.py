import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class E5Retriever:
    """
    Dense retriever for MS MARCO passages using E5.

    Important:
    - E5 expects query inputs prefixed with 'query: '
    - E5 expects document inputs prefixed with 'passage: '
    - embeddings are normalized, so inner product = cosine similarity
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        corpus_name: str = "msmarco-passage",
        index_dir: str = "data/e5_index",
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.corpus_name = corpus_name
        self.index_dir = Path(index_dir)
        self.batch_size = batch_size
        self.device = device

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name, device=device)
        self.dataset = ir_datasets.load(corpus_name)
        self.doc_store = self.dataset.docs_store()

        safe_name = model_name.replace("/", "__")
        self.index_path = self.index_dir / f"{safe_name}.faiss"
        self.meta_path = self.index_dir / f"{safe_name}.json"

        self.index = None

    def _encode_passages(self, passages: List[str]) -> np.ndarray:
        inputs = [f"passage: {text}" for text in passages]
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        inputs = [f"query: {text}" for text in queries]
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def build_index(self, rebuild: bool = False, max_docs: Optional[int] = None):
        """
        Build a FAISS index over MS MARCO passages.
        """
        if self.index_path.exists() and not rebuild:
            self.index = faiss.read_index(str(self.index_path))
            return

        test_vec = self._encode_passages(["hello world"])
        dim = test_vec.shape[1]

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

        doc_ids_batch = []
        texts_batch = []
        total = 0

        for doc in tqdm(self.dataset.docs_iter(), desc="Encoding corpus"):
            if max_docs is not None and total >= max_docs:
                break

            doc_ids_batch.append(np.int64(doc.doc_id))
            texts_batch.append(doc.text)
            total += 1

            if len(texts_batch) >= self.batch_size * 8:
                embs = self._encode_passages(texts_batch)
                ids = np.array(doc_ids_batch, dtype=np.int64)
                index.add_with_ids(embs, ids)
                doc_ids_batch.clear()
                texts_batch.clear()

        if texts_batch:
            embs = self._encode_passages(texts_batch)
            ids = np.array(doc_ids_batch, dtype=np.int64)
            index.add_with_ids(embs, ids)

        faiss.write_index(index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "corpus_name": self.corpus_name,
                    "size": int(index.ntotal),
                    "max_docs": max_docs,
                },
                f,
                indent=2,
            )

        self.index = index

    def load_index(self):
        if self.index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"No FAISS index found at {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict]:
        self.load_index()

        q_emb = self._encode_queries([query])
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        for doc_id, score in zip(ids[0], scores[0]):
            if doc_id == -1:
                continue

            doc = self.doc_store.get(str(int(doc_id)))
            if doc is None:
                continue

            results.append(
                {
                    "docid": str(int(doc_id)),
                    "score": float(score),
                    "contents": doc.text,
                }
            )
        return results

    def batch_retrieve(self, queries: Dict[str, str], top_k: int = 100) -> Dict[str, List[Dict]]:
        self.load_index()
        results = {}

        for qid, query in tqdm(queries.items(), desc="E5 retrieval"):
            results[qid] = self.retrieve(query, top_k=top_k)

        return results

    def batch_retrieve_pyserini_style(self, queries: Dict[str, str], top_k: int = 100) -> Dict[str, Dict]:
        """
        Save output in a structure similar to data/dl19_pyserini_bm25.json
        """
        raw = {}
        dense_results = self.batch_retrieve(queries, top_k=top_k)

        for qid, docs in dense_results.items():
            raw[qid] = {
                "query": queries[qid],
                "passages": [
                    {
                        "pid": doc["docid"],
                        "text": doc["contents"],
                        "score": doc["score"],
                    }
                    for doc in docs
                ],
            }

        return raw
