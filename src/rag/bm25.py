import re

from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

from tqdm import tqdm

from .chunking import Chunk


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "of", "to", "in", "on", "for", "and", "or",
    "what", "which", "how", "does", "do"
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())

    tokens = []

    for word in words:
        if word not in STOPWORDS:
            tokens.append(word)

        if "_" in word:
            for part in word.split("_"):
                if part and part not in STOPWORDS:
                    tokens.append(part)

    return tokens


class BM25Search:
    def __init__(self, cache: bool = False) -> None:
        self.chunks: list[Chunk] = []
        self.bm25 = None
        self.use_cache = cache

        self.cache_path = Path("data/processed/cache_query.pkl")
        self.cache: dict[tuple[str, int], list[Chunk]]
        if cache and self.cache_path.exists():
            with open(self.cache_path, "rb") as file:
                self.cache = pickle.load(file)
        else:
            self.cache = {}

    def index(self, chunks: list[Chunk]) -> None:
        cache_path = Path("data/processed/cache_index.pkl")
        chunks_path = Path("data/processed/chunks.json")

        self.chunks = chunks

        if (
            self.use_cache
            and cache_path.exists()
            and chunks_path.exists()
            and cache_path.stat().st_mtime >= chunks_path.stat().st_mtime
        ):
            with open(cache_path, "rb") as file:
                self.bm25 = pickle.load(file)
            return

        corpus = [
            tokenize(chunk.content)
            for chunk in tqdm(chunks, desc="Tokenizing chunks", unit="chunk")
        ]
        self.bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, "wb") as file:
                pickle.dump(self.bm25, file)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        if self.bm25 is None or top_k <= 0 or not tokenize(query):
            return []
        key = (query, top_k)
        if self.use_cache and key in self.cache:
            print("CACHE HIT")
            return self.cache[key]

        scores = self.bm25.get_scores(tokenize(query))

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = [
            self.chunks[i]
            for i in ranked[:top_k]
            if scores[i] > 0
        ]

        if self.use_cache:
            self.cache[key] = results

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as file:
                pickle.dump(self.cache, file)

        return results
