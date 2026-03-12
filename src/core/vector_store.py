import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import faiss
import numpy as np

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512
DEFAULT_STORE_DIR = os.path.join(os.path.expanduser("~"), ".tpv_golf")
DEFAULT_STORE_PATH = os.path.join(DEFAULT_STORE_DIR, "vectors")


@dataclass
class SearchResult:
    path: str
    score: float
    index_id: int


@dataclass
class VectorStore:
    index: faiss.IndexFlatIP | None = None
    metadata: list[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    def build_from_embeddings(self, items: list[dict]):
        """Build index from list of {path, embedding} dicts."""
        if not items:
            return
        vectors = np.stack([item["embedding"] for item in items]).astype(np.float32)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.metadata = [{"path": item["path"]} for item in items]
        log.info("Built FAISS index with %d vectors", self.index.ntotal)

    def build_from_folder(
        self,
        folder: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        from core.embedder import batch_embed_folder

        items = batch_embed_folder(folder, progress_callback, cancel_check)
        self.build_from_embeddings(items)

    def search(self, query_vector: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self.is_empty():
            return []
        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append(SearchResult(
                path=meta.get("path", ""),
                score=float(score),
                index_id=int(idx),
            ))
        return results

    def save(self, path: str):
        """Save index and metadata to disk. Creates path.faiss and path.json."""
        if self.index is None:
            return
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".json", "w") as f:
            json.dump(self.metadata, f)
        log.info("Saved vector store to %s (.faiss + .json)", path)

    def load(self, path: str):
        """Load index and metadata from disk."""
        faiss_path = path + ".faiss"
        json_path = path + ".json"
        if not os.path.exists(faiss_path) or not os.path.exists(json_path):
            raise FileNotFoundError(f"Vector store files not found at {path}")
        self.index = faiss.read_index(faiss_path)
        with open(json_path) as f:
            self.metadata = json.load(f)
        log.info("Loaded vector store: %d vectors", self.index.ntotal)

    # ---- Incremental operations for admin panel ----

    def get_all_entries(self) -> list[dict]:
        """Return list of {index_id, path} for every vector in the store."""
        return [
            {"index_id": i, "path": m.get("path", "")}
            for i, m in enumerate(self.metadata)
        ]

    def add_single_image(self, image_path: str) -> bool:
        """Embed one image and append to the index. Returns True if successful."""
        from core.embedder import get_embedding
        emb = get_embedding(image_path)
        if emb is None:
            return False
        self.add_embeddings([{"path": image_path, "embedding": emb}])
        return True

    def add_embeddings(self, items: list[dict]):
        """Append {path, embedding} items to an existing (or new) index."""
        if not items:
            return
        vectors = np.stack([it["embedding"] for it in items]).astype(np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)
        self.metadata.extend({"path": it["path"]} for it in items)
        log.info("Added %d vectors (total: %d)", len(items), self.index.ntotal)

    def remove_by_indices(self, indices_to_remove: set[int]):
        """Remove vectors at the given index positions by rebuilding the index."""
        if self.index is None or not indices_to_remove:
            return
        n = self.index.ntotal
        all_vectors = faiss.rev_swig_ptr(self.index.get_xb(), n * EMBEDDING_DIM)
        all_vectors = np.reshape(all_vectors, (n, EMBEDDING_DIM)).copy()

        keep_mask = [i not in indices_to_remove for i in range(n)]
        kept_vectors = all_vectors[keep_mask]
        kept_metadata = [m for i, m in enumerate(self.metadata) if i not in indices_to_remove]

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if len(kept_vectors) > 0:
            self.index.add(kept_vectors.astype(np.float32))
        self.metadata = kept_metadata
        log.info("Removed %d vectors (remaining: %d)", len(indices_to_remove), self.index.ntotal)

    def has_path(self, path: str) -> bool:
        return any(m.get("path") == path for m in self.metadata)
