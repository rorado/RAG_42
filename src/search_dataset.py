
import json
from pathlib import Path
from .main import Retriever
from .classes import (
    MinimalSearchResults,
    InvalidFormatError,
    MinimalSource,
    StudentSearchResults,
)



def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
    method: str = "lexical",
):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    retriever = Retriever(method)
    search_results = []

    for item in data["rag_questions"]:
        if "question" not in item:
            raise InvalidFormatError("Missing question")

        if "question_id" not in item:
            raise InvalidFormatError("Missing question_id")

        question = item["question"]

        chunks = retriever.search(question, k)

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