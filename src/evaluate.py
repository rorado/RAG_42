import json
from pathlib import Path

from .classes import StudentSearchResults


def is_correct_source(expected, retrieved):
    if expected["file_path"] != retrieved.file_path:
        return False

    start = max(
        expected["first_character_index"],
        retrieved.first_character_index,
    )

    end = min(
        expected["last_character_index"],
        retrieved.last_character_index,
    )

    intersection = max(0, end - start)

    union = (
        max(
            expected["last_character_index"],
            retrieved.last_character_index,
        )
        - min(
            expected["first_character_index"],
            retrieved.first_character_index,
        )
    )

    if union <= 0:
        return False

    iou = intersection / union

    return iou >= 0.05


def run_evaluate(
    student_search_results_path: str,
    dataset_path: str,
):
    results = StudentSearchResults.model_validate_json(
        Path(student_search_results_path).read_text(encoding="utf-8")
    )

    ground_truth = json.loads(
        Path(dataset_path).read_text(encoding="utf-8")
    )

    questions = {
        str(item["question_id"]): item
        for item in ground_truth["rag_questions"]
    }

    total_recall = 0.0
    evaluated = 0

    for item in results.search_results:
        question = questions.get(str(item.question_id))

        if question is None:
            continue

        correct_sources = question["sources"]

        if not correct_sources:
            continue

        retrieved_sources = item.retrieved_sources[:results.k]

        matches = 0

        for expected in correct_sources:
            if any(
                is_correct_source(expected, retrieved)
                for retrieved in retrieved_sources
            ):
                matches += 1

        recall = matches / len(correct_sources)

        total_recall += recall
        evaluated += 1

    if evaluated == 0:
        raise ValueError("No questions could be evaluated.")

    average_recall = total_recall / evaluated

    # print(f"Questions evaluated: {evaluated}")
    # print(f"Recall@{results.k}: {average_recall:.2%}")

    return average_recall