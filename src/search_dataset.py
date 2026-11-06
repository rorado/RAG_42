
import json
from pathlib import Path

from bm25 import BM25Search

from .classes import (
    MinimalSearchResults,
    InvalidFormatError,
    MinimalSource,
    StudentSearchResults,
)
from .semantic_search import Semantic
from .chunk_loader import load_chunks


def search_dataset(
    dataset_path: str,
    save_directory: str,
    bm25: BM25Search = None,
    k: int = 5,
    method: str = "lexical",
):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    search_results = []

    if method == "semantic":
        all_chunks = load_chunks()

        semantic = Semantic()
        semantic.load()

        print("Chunks loaded:", len(all_chunks))
        print("Embeddings loaded:", len(semantic.embeddings))

        if len(all_chunks) != len(semantic.embeddings):
            raise ValueError(
                "Chunks and embeddings do not match"
            )

    elif method == "lexical":
        if bm25 is None:
            raise ValueError("BM25 index is required for lexical search")

    else:
        raise ValueError(f"Unknown search method: {method}")

    for item in data["rag_questions"]:
        if "question" not in item:
            raise InvalidFormatError("Missing question")

        if "question_id" not in item:
            raise InvalidFormatError("Missing question_id")

        question = item["question"]

        if method == "lexical":
            chunks = bm25.search(question, top_k=k)

        else:
            chunks = semantic.search(
                query=question,
                chunks=all_chunks,
                top_k=k,
            )

        sources = [
            MinimalSource(
                file_path=chunk.path,
                first_character_index=chunk.start,
                last_character_index=chunk.end,
            )
            for chunk in chunks
        ]

        search_results.append(
            MinimalSearchResults(
                question_id=item["question_id"],
                question=question,
                retrieved_sources=sources,
            )
        )

    output = StudentSearchResults(
        search_results=search_results,
        k=k,
    )

    output_path = Path(save_directory) / Path(dataset_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path
