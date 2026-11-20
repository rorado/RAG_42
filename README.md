*This project has been created as part of the 42 curriculum by sohrich.*

# RAG against the Machine — RAG_42

## Description

RAG_42 is a **retrieval-augmented generation (RAG)** system built for the 42 project *RAG against the Machine*. It answers questions about the **vLLM 0.10.1** source code and documentation by retrieving relevant passages and supplying them as context to a language model. The objective is to produce answers grounded in the indexed repository instead of relying on the model's general knowledge alone.

The project combines lexical and semantic search, supports incremental indexing, and exposes question-answering functionality through Python and a FastAPI interface.

## System architecture

```text
vLLM source files (.py, .md)
          |
          v
   File discovery + SHA-256 change detection
          |
          v
   Chunking (Python AST-aware / text chunks)
          |
          v
   Persisted chunks and file-hash metadata
          |
          v
   +----------------------+----------------------+
   |                                             |
   v                                             v
BM25 lexical search                       Semantic search
   |                                             |
   +----------------------+----------------------+
                          |
                          v
               Reciprocal Rank Fusion (RRF)
                          |
                          v
                 Top-k source chunks
                          |
                          v
               Prompt with source context
                          |
                          v
                 Qwen3-0.6B LLM
                          |
                          v
                    Grounded answer
```

**Indexing:** Source files are discovered recursively, segmented into chunks, and saved for later retrieval. SHA-256 hashes help identify added or modified files, while removed files can be excluded from the saved index.

**Retrieval:** A user query is searched using BM25 and semantic similarity. Results are combined using RRF and the highest-ranked chunks are selected as evidence.

**Generation:** Retrieved chunk text is passed to the language model alongside the question. If retrieval finds no relevant sources, the application can return a no-sources response rather than generating an unsupported answer.

**API:** A FastAPI application initializes the retrieval index and LLM during its lifespan startup and exposes `/search` and `/answer` endpoints.

## Chunking strategy

The indexer processes `.py` and `.md` files using different approaches:

- **Python files:** Uses AST to identify top-level statement boundaries and split code into chunks without exceeding the maximum size. If no suitable AST boundary is found, it splits at the last available newline or at the size limit.

- **Markdown files:** Splits text into size-limited chunks, preferring newlines within the last 30% of each chunk, then spaces, and finally the maximum size limit.


## Retrieval method and ranking

### BM25

BM25 ranks chunks using query-term frequency, term rarity across the corpus, and document-length normalization. It is effective when questions mention exact identifiers, function names, or technical vocabulary present in the source.

### Semantic search

Semantic retrieval compares vector representations of a query and candidate chunks to find conceptually related content, even when the wording differs. This complements BM25's exact-term matching.

### Reciprocal Rank Fusion (RRF)

The two result lists are merged using reciprocal rank fusion:

$$
\operatorname{RRF}(d) =
\sum_{r \in \{\mathrm{BM25},\mathrm{Semantic}\}}
\frac{1}{60 + \operatorname{rank}_r(d)}
$$

A chunk appearing near the top of both rankings receives a higher combined score. Chunks are identified by their **(path, start, end)** tuple to avoid counting the same source segment as distinct results. The combined results are sorted by RRF score and truncated to the requested `top_k`.

## Instructions

### Requirements

- Python and [`uv`](https://docs.astral.sh/uv/) installed
- Project dependencies available through the repository's `pyproject.toml` / lockfile
- The vLLM 0.10.1 source tree available at the configured input directory (the example path is `data/raw/vllm-0.10.1`)
- Any model weights or runtime dependencies required by the LLM and semantic search components

### Installation

From the repository root:

```bash
uv sync
```


### Running the application

Run the following command to see the available options:

```bash
uv run -m src
```


For the FastAPI interface, start the module that defines the `app = FastAPI(...)` instance. For example, **if** that file is `src/api.py`:

```bash
uv run uvicorn local_http.api:app --host 127.0.0.1 --port 8000
```

### Example usage

Search for relevant code chunks:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does vLLM manage the KV cache?","top_k":5}'
```

Ask for a grounded answer:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is continuous batching in vLLM?","top_k":5}'
```

The `/search` response contains the query and source chunks with `path`, `start`, `end`, and `content`. The `/answer` response contains a generated answer, or a no-sources message when retrieval is empty.

## Performance Analysis

The system was evaluated using the 42 RAG evaluation harness on two private datasets containing 100 documentation questions and 100 code questions.

### Retrieval Accuracy

| Evaluation setting | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| Documentation | 61.0% | 80.0% | 83.0% | 88.0% |
| Code | 40.0% | 58.0% | 64.0% | 70.0% |

Both datasets passed the required Recall@5 thresholds:

- **Documentation:** 83% (required: 80%)
- **Code:** 64% (required: 50%)

These results correspond to the retrieval configuration used during evaluation. Separate BM25-only and semantic-only benchmark results have not been reported.


### Observations

Documentation retrieval achieved higher Recall@5 than code retrieval (83% versus 64%). This suggests that retrieving relevant source-code chunks is more challenging for the evaluated system.

Both retrieval accuracy and execution speed satisfied the project's evaluation requirements.

## Design decisions

| Decision | Reason |
| --- | --- |
| AST-aware Python chunking | Preserves code structure better than arbitrary fixed-width splitting |
| Separate Markdown chunking | Documentation text does not have Python syntax boundaries |
| BM25 + semantic retrieval | Combines precise keyword matching with conceptual similarity |
| RRF rather than raw-score addition | BM25 and vector-similarity scores use different scales; rank fusion avoids direct score calibration |
| SHA-256 file hashes | Detects source changes for incremental re-indexing |
| Persisted chunks and metadata | Avoids reprocessing unchanged files on every run |
| Pydantic request/data models | Provides validation and explicit data shapes |
| FastAPI lifespan initialization | Creates shared retrieval/model resources once at application startup |
| Small Qwen3-0.6B generator | Fits the project's specified model constraints; places emphasis on retrieval quality and grounding |

## Challenges faced

**Preserving Python syntax during chunking.** Splitting solely by character count can cut through functions or statements. The solution was to use AST-derived boundaries and a fallback for oversized regions.

**Balancing retrieval methods.** BM25 favors exact vocabulary while semantic search can recover differently phrased concepts. Combining ranked lists with RRF reduces dependence on either method alone.

**Avoiding unnecessary full rebuilds.** Reprocessing an entire repository after small edits wastes time. Saved SHA-256 hashes, file-deletion detection, and chunk metadata support incremental updates. The saved `chunks.json` modification timestamp is used to check consistency with the hash metadata.

**Meeting evaluation latency requirements.** Index construction, model loading, and query retrieval have different costs. Shared startup initialization and warm-retrieval measurements help distinguish one-time setup overhead from per-question performance.

**Grounding small-model answers.** A compact language model may omit details or hallucinate. Supplying relevant retrieved context and handling empty retrieval results reduces this risk, although it does not eliminate it.

## Resources

### References

- [Python `ast` parcing](https://youtu.be/OjPT15y2EpE?si=o1f1VOVQQlMNybzD) — parsing and inspecting Python source structure
- [BM25 vs TF-IDF](https://youtu.be/0s0aBl2U3vg?si=v8guBJf87FtkZuKJ) — SHA-256 and file hashing

### Use of AI

AI assistance was used for **learning and explanation**, including Python generators and iterators, AST-based chunking, BM25 ranking, reciprocal rank fusion, SHA-256 hashing, FastAPI lifespan behavior, and debugging concepts. AI was also used to **organize this README**. 

