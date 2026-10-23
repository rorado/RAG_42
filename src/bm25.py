import re
from rank_bm25 import BM25Okapi
from chunking import Chunk


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

    def __init__(self):
        self.chunks = []
        self.bm25 = None

    def index(self, chunks: list[Chunk]):
        self.chunks = chunks

        corpus = [tokenize(chunk.content) for chunk in chunks]

        self.bm25 = (
            BM25Okapi(corpus, k1=1.5, b=0.75)
            if corpus and any(corpus)
            else None
        )

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        if self.bm25 is None or top_k <= 0:
            return []

        tokens = tokenize(query)

        if not tokens:
            return []

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

# texts = [
#     "",
#     "user login authentication",
#     "database connection postgres",
#     "python cache management",
#     "network request response",
# ]

# chunks = [
#     Chunk(
#         id=str(i),
#         content=text,
#         size=len(text),
#         path="test.py",
#         start=0,
#         end=len(text),
#     )
#     for i, text in enumerate(texts)
# ]

# engine = BM25Search(chunks)

# results = engine.search("cache", top_k=2)

# for chunk in results:
#     print(chunk.content)
