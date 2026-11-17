import re
from collections import OrderedDict

from rank_bm25 import BM25Okapi

from .cache import QueryCache

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

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.query_cache: QueryCache | None = None
        self._results: OrderedDict[
            tuple[str, int], list[int]
        ] = OrderedDict()

    def index(self, chunks: list[Chunk]) -> None:
        self._results.clear()
        self.query_cache = None
        self.chunks = chunks

        corpus = [tokenize(chunk.content) for chunk in chunks]

        self.bm25 = (
            BM25Okapi(corpus, k1=1.5, b=0.75)
            if corpus and any(corpus)
            else None
        )

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Search using memory and optional persistent result caches."""
        if self.bm25 is None or top_k <= 0 or not tokenize(query):
            return []

        key = (query, top_k)
        indices = self._results.get(key)
        if indices is None and self.query_cache is not None:
            indices = self.query_cache.get(query, top_k, len(self.chunks))

        if indices is None:
            scores = self.bm25.get_scores(tokenize(query))
            ranked = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True,
            )
            indices = [i for i in ranked[:top_k] if scores[i] > 0]
            if self.query_cache is not None:
                self.query_cache.put(query, top_k, indices)

        self._results[key] = indices
        self._results.move_to_end(key)
        if len(self._results) > 1000:
            self._results.popitem(last=False)
        return [self.chunks[i].model_copy() for i in indices]

    def search_cache(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Keep the original cached-search entry point available."""
        return self.search(query, top_k)
