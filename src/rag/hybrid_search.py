from .bm25 import BM25Search
from .chunking import Chunk
from .semantic_search import Semantic


class HybridSearch:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Search,
        semantic: Semantic,
    ):
        self.chunks = chunks
        self.bm25 = bm25
        self.semantic = semantic

    def rrf(self, query: str, top_k: int = 5):
        if not query.strip() or top_k <= 0:
            return []

        candidate_k = max(20, top_k)

        bm25_res = self.bm25.search(query, top_k=candidate_k)
        semantic_res = self.semantic.search(
            query, self.chunks, top_k=candidate_k
        )

        rrf_res = {}
        chunk_map = {}

        for results in [bm25_res, semantic_res]:
            for rank, chunk in enumerate(results, start=1):
                key = (chunk.path, chunk.start, chunk.end)
                if key not in rrf_res:
                    rrf_res[key] = 0

                rrf_res[key] += 1 / (60 + rank)
                chunk_map[key] = chunk

        ranked = sorted(
            rrf_res,
            key=rrf_res.get,
            reverse=True,
        )

        return [chunk_map[key] for key in ranked[:top_k]]
