from pathlib import Path
from .classes import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
    method: str = "lexical",
    cache: bool = False,
) -> Path:
    from .main import Retriever

    if k <= 0:
        raise ValueError("k must be greater than 0")

    dataset = RagDataset.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8")
    )
    retriever = Retriever(method, cache=cache)
    search_results = []

    for item in dataset.rag_questions:
        chunks = retriever.search(item.question, k)

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
                question_id=item.question_id,
                question=item.question,
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
