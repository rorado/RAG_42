from pathlib import Path

from .classes import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .llm import LLM


def run_answer_dataset(
    dataset_path: str,
    save_directory: str,
):
    data = StudentSearchResults.model_validate_json(
        Path(dataset_path).read_text(encoding="utf-8")
    )

    llm = LLM()
    answers: list[MinimalAnswer] = []

    for item in data.search_results:
        question = item.question
        context_parts = []

        for source in item.retrieved_sources:
            content = Path(source.file_path).read_text(encoding="utf-8")

            chunk_text = content[
                source.first_character_index:source.last_character_index
            ]

            context_parts.append(chunk_text)

        # Combine all retrieved chunks
        context = "\n\n".join(context_parts)

        # Generate answer using Qwen
        if context.strip():
            answer = llm.answer(question, context)
        else:
            answer = "Not enough information in the provided context."

        answers.append(
            MinimalAnswer(
                question_id=item.question_id,
                question=question,
                retrieved_sources=item.retrieved_sources,
                answer=answer,
            )
        )

    output = StudentSearchResultsAndAnswer(
        search_results=answers,
        k=data.k,
    )

    output_path = Path(save_directory) / Path(dataset_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    return output_path
