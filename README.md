# RAG against the machine

## Bonus: caching

Lexical retrieval caches the built BM25 index and ranked query results on disk.
Caching is enabled by default for `search`, `search_dataset`, and `answer`, and
for lexical retrieval in the local HTTP API. Hybrid retrieval also reuses the
lexical index and lexical query results.

```bash
uv run python -m src search "How does vLLM schedule requests?" --k 5
# Repeat in a new process to reuse both caches.
uv run python -m src search "How does vLLM schedule requests?" --k 5
# Bypass persistent caching for comparison.
uv run python -m src search "How does vLLM schedule requests?" --k 5 --cache=False
make cache_query QUERY='How does vLLM schedule requests?'
```

`data/processed/cache_index.pkl` stores the tokenized BM25 index, avoiding
retokenization and index construction on subsequent CLI starts.
`data/processed/query_cache.sqlite3` stores ranked chunk positions keyed by the
exact query, requested k, and index generation. Both memory and disk query caches
retain at most 1000 entries using least recently used eviction. Empty results
are cached too. Returned chunks are copied so callers cannot mutate cache hits.

Rewriting `chunks.json` through full or incremental indexing invalidates both
persistent caches automatically. The signature includes the resolved path, file
identity, size, and nanosecond modification/change timestamps, plus a cache
format version. Increment `CACHE_VERSION` in `src/rag/cache.py` after changing
tokenization or ranking behavior. Missing or corrupt cache data falls back to
normal retrieval; cache write failures do not prevent searching.

`make clean` removes these generated caches along with the processed index.
Cache files are local generated artifacts excluded from Git; the pickle index
must only come from your own trusted project runs. Semantic embeddings already
persist separately; semantic query encoding and final hybrid rankings are not
cached by this feature.
