import re
from rank_bm25 import BM25Okapi
from chunking import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Search:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

        corpus = [tokenize(chunk.content) for chunk in chunks]
        print(corpus)

        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        if self.bm25 is None or top_k <= 0:
            return []

        tokens = tokenize(query)

        scores = self.bm25.get_scores(tokens)

        indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.chunks[i]
            for i in indices[:top_k]
            if scores[i] > 0
        ]
