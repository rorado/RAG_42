
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        self.embeddings = None

    def index(self, chunks, path="data/processed/embeddings.npy"):
        texts = [chunk.content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.embeddings = embeddings

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    def load(self, path="data/processed/embeddings.npy"):
        self.embeddings = np.load(path)

    def search(self, query, chunks, top_k=20):
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        if len(chunks) != len(self.embeddings):
            raise ValueError("Chunks and embeddings do not match")

        if not chunks or top_k <= 0 or not query.strip():
            return []

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]

        return [chunks[int(i)] for i in indices]



text = "How to free GPU memory?"



