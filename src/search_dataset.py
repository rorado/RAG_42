import json
from pathlib import Path
from bm25 import BM25Search
from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int

class ParceError(Exception):
    pass
def search_dataset(dataset_path: str, save_directory:str, bm25:BM25Search, k: int=5):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    search_results:list[MinimalSearchResults] = []

    for item in data["rag_questions"]:
        if "question" not in item:
            raise ParceError(f"question not exit in {dataset_path}") 
        elif "question_id" not in item:
            raise ParceError(f"question not exit in {dataset_path}") 
        question = item["question"]

        chunks = bm25.search(question, top_k=k)

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
                retrieved_sources=sources
            )
        )

    output = StudentSearchResults(
        search_results = search_results,
        k=k,
    )
    output_path = Path(save_directory) / Path(dataset_path).name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    return output_path