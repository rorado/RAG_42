from pathlib import Path
from .chunking import Chunk
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class Semantic:
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        self.embeddings: NDArray[np.float32] | None = None

    def index(
        self, chunks: list[Chunk],
        path: str = "data/processed/embeddings.npy",
    ) -> None:
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

    def load(self, path: str = "data/processed/embeddings.npy") -> None:
        self.embeddings = np.load(path)

    def search(
        self, query: str, chunks: list[Chunk], top_k: int = 20,
    ) -> list[Chunk]:
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        if len(chunks) != len(self.embeddings):
            raise ValueError("Chunks and embeddings do not match")

        if len(chunks) == 0 or top_k <= 0 or not query.strip():
            return []

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]

        return [chunks[int(i)] for i in indices]
