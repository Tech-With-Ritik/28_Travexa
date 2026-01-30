import faiss
import numpy as np


class FAISSStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def _fix_embedding(self, emb):
        """
        Converts ANY embedding shape into (dim,)
        Handles:
        - (dim,)
        - (1, dim)
        - (tokens, hidden)
        - (1, tokens, hidden)
        """

        arr = np.array(emb, dtype="float32")

        # Case 1: (1, tokens, hidden)
        if arr.ndim == 3:
            arr = arr[0]              # -> (tokens, hidden)
            arr = arr.mean(axis=0)    # -> (hidden,)

        # Case 2: (tokens, hidden)
        elif arr.ndim == 2:
            arr = arr.mean(axis=0)    # -> (hidden,)

        # Case 3: (1, dim)
        elif arr.ndim == 1:
            pass

        else:
            raise ValueError(f"Unsupported embedding shape: {arr.shape}")

        # Now arr is (hidden,)
        if arr.shape[0] < self.dim:
            raise ValueError(f"Embedding too small: {arr.shape}")

        # Trim or project to dim
        arr = arr[:self.dim]

        return arr

    def add(self, embeddings, metadatas):
        if not embeddings or not metadatas:
            return

        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        fixed_vectors = []
        for emb in embeddings:
            fixed_vectors.append(self._fix_embedding(emb))

        vectors = np.vstack(fixed_vectors).astype("float32")

        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0 or not self.metadata:
            return []

        q = self._fix_embedding(query_embedding).reshape(1, -1)

        D, I = self.index.search(q, k)

        results = []
        for rank, meta_idx in enumerate(I[0]):
            if meta_idx == -1:
                continue
            if meta_idx < 0 or meta_idx >= len(self.metadata):
                continue
            results.append((self.metadata[meta_idx], float(D[0][rank])))

        return results
