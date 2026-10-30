
import os
import json
from pathlib import Path

from rich.console import Console

from .chunking import chunk_file, chunk_file_py, Chunk, store_chunks
from .bm25 import BM25Search
from .llm import LLM
from .search_dataset import search_dataset as run_search_dataset
from .answer_dataset import run_answer_dataset

MAX_CHUNK_SIZE = 2000
VLLM_PATH = "data/raw"

console = Console()


def search_in_vllm_files(
    dir: str = VLLM_PATH,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[Chunk]:

    chunks_list: list[Chunk] = []

    with os.scandir(dir) as entries:
        for en in entries:

            if en.is_dir():
                chunks_list.extend(
                    search_in_vllm_files(en.path, max_chunk_size)
                )

            elif en.is_file() and en.name.endswith(".py"):
                with open(en.path, "r", encoding="utf-8") as file:
                    content = file.read()

                chunks_list.extend(
                    chunk_file_py(content, en.path, max_chunk_size)
                )

            elif en.is_file() and en.name.endswith(".md"):
                with open(en.path, "r", encoding="utf-8") as file:
                    content = file.read()

                chunks_list.extend(
                    chunk_file(content, en.path, max_chunk_size)
                )

    return chunks_list


def load_chunks(folder_path: str = "data/processed") -> list[Chunk]:

    file_path = Path(folder_path) / "chunks.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [Chunk.model_validate(item) for item in data]


def create_bm25() -> BM25Search:

    chunks = load_chunks()

    bm25 = BM25Search()
    bm25.index(chunks)

    return bm25


# Command: index
def index(max_chunk_size: int = 2000):

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")

    chunks = search_in_vllm_files(VLLM_PATH, max_chunk_size)

    store_chunks(chunks, "data/processed")

    console.print(
        f"[bold green]Indexed {len(chunks)} chunks[/bold green]"
    )


# Command: search
def search(query: str, k: int = 5):

    if k <= 0:
        raise ValueError("k must be greater than 0")

    bm25 = create_bm25()

    results = bm25.search(query, k)

    for chunk in results:
        console.print(f"\n[green]Path:[/green] {chunk.path}")
        console.print(f"Start: {chunk.start}")
        console.print(f"End: {chunk.end}")
        console.print(f"Content:\n{chunk.content}")


# Command: search_dataset
def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
):

    if k <= 0:
        raise ValueError("k must be greater than 0")

    bm25 = create_bm25()

    output_path = run_search_dataset(
        dataset_path=dataset_path,
        save_directory=save_directory,
        bm25=bm25,
        k=k,
    )

    console.print(f"[green]Saved to:[/green] {output_path}")


# Command: answer
def answer(query: str, k: int = 5):

    if k <= 0:
        raise ValueError("k must be greater than 0")

    bm25 = create_bm25()

    results = bm25.search(query, k)

    if not results:
        console.print("[yellow]No sources found[/yellow]")
        return

    context = "\n\n".join(
        chunk.content for chunk in results
    )

    llm = LLM()

    response = llm.answer(query, context)

    console.print("[bold green]Answer:[/bold green]")
    print(response)


# Command: answer_dataset
def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    k: int = 5,
):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    bm25 = create_bm25()
    
    output_path = run_answer_dataset(
        dataset_path=student_search_results_path,
        save_directory=save_directory,
        bm25=bm25,
        k=k,
    )
    
    console.print(f"[green]Saved to:[/green] {output_path}")

