
import os
from pathlib import Path
from rich.console import Console
from bm25 import BM25Search
from .chunking import chunk_file, chunk_file_py, Chunk, store_chunks
from .llm import LLM
from .search_dataset import search_dataset as run_search_dataset
from .answer_dataset import run_answer_dataset
from .evaluate import run_evaluate
from .semantic_search import Semantic
from .incremental import detect_changes, save_hashes
from pathlib import Path
import pickle
from .hybrid_search import HybridSearch
from .semantic_search import Semantic
from .chunk_loader import load_chunks



MAX_CHUNK_SIZE = 2000
VLLM_PATH = "data/raw"
WITH_CACHE = False
console = Console()




class Retriever:
    def __init__(self, method="lexical"):
        if method not in ("lexical", "semantic", "hybrid"):
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.bm25 = None
        self.semantic = None
        self.hybrid = None
        self.chunks = None

        if method in ("lexical", "hybrid"):
            self.bm25 = create_bm25()

        if method in ("semantic", "hybrid"):
            self.chunks = load_chunks()
            self.semantic = Semantic()
            self.semantic.load()

            if len(self.chunks) != len(self.semantic.embeddings):
                raise ValueError("Chunks and embeddings do not match")

        if method == "hybrid":
            self.hybrid = HybridSearch(
                chunks=self.chunks,
                bm25=self.bm25,
                semantic=self.semantic,
            )

    def search(self, query: str, k: int = 5):
        if self.method == "lexical":
            return self.bm25.search(query, top_k=k)

        if self.method == "semantic":
            return self.semantic.search(
                query=query,
                chunks=self.chunks,
                top_k=k,
            )

        return self.hybrid.rrf(query=query, top_k=k)





def incremental_index(
    directory: str = "data/raw/vllm-0.10.1",
    max_chunk_size: int = MAX_CHUNK_SIZE,
):
    chunks_path = Path("data/processed/chunks.json")

    changed_files, deleted_files, new_hashes = detect_changes(directory)

    if not chunks_path.exists():
        print("Chunks do not exist yet. Creating the index...")
        index(max_chunk_size)
        save_hashes(new_hashes)
        return


    if not changed_files and not deleted_files:
        print("No changes detected")
        return

    old_chunks = load_chunks()

    affected = {
        str(path) for path in changed_files
    } | set(deleted_files)

    chunks = [
        chunk
        for chunk in old_chunks
        if chunk.path not in affected
    ]

    for path in changed_files:
        content = path.read_text(encoding="utf-8")

        if path.suffix == ".py":
            chunks.extend(
                chunk_file_py(
                    content,
                    str(path),
                    max_chunk_size,
                )
            )

        elif path.suffix == ".md":
            chunks.extend(
                chunk_file(
                    content,
                    str(path),
                    max_chunk_size,
                )
            )

    # Save updated chunks.
    store_chunks(chunks, "data/processed")

    # Save hashes only after successful chunk indexing.
    save_hashes(new_hashes)

    print(f"Changed/new files: {len(changed_files)}")
    print(f"Deleted files: {len(deleted_files)}")
    print(f"Total chunks: {len(chunks)}")


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


def index(max_chunk_size: int = 2000, store_path = "data/processed"):
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")

    chunks = search_in_vllm_files(VLLM_PATH, max_chunk_size)

    store_chunks(chunks, store_path)

    console.print(
        f"[bold green]Indexed {len(chunks)} chunks[/bold green]"
    )


def index_embeddings(max_chunk_size: int = 2000):
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than 0")
    
    chunks = search_in_vllm_files(VLLM_PATH, max_chunk_size)
    semantic = Semantic()

    semantic.index(chunks)

    console.print(
        f"[bold green]Indexed {len(chunks)} chunks semanticlly[/bold green]"
    )

def create_bm25(cache=False) -> BM25Search:
    cache_path = Path("data/processed/cache_index.pkl")
    chunks_path = Path("data/processed/chunks.json")

    if (cache_path.exists() and cache
        and cache_path.stat().st_mtime_ns >= chunks_path.stat().st_mtime_ns): 
        with open(cache_path, "rb") as file:
            return(pickle.load(file))


    chunks = load_chunks()

    bm25 = BM25Search()
    bm25.index(chunks)

    with open(cache_path, "wb") as file:
        pickle.dump(bm25, file)
    return bm25


def search(
    query: str,
    k: int = 5,
    # cache: bool = False,
    method: str = "lexical",
):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if method not in ("lexical", "semantic", "hybrid"):
        raise ValueError(f"Unknown search method: {method}")

    retriever = Retriever(method)
    results = retriever.search(query, k)

    for chunk in results:
        console.print(f"\n[green]Path:[/green] {chunk.path}")
        console.print(f"Start: {chunk.start}")
        console.print(f"End: {chunk.end}")


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 5,
    method:str = "lexical"
):

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if method not in ("lexical", "semantic", "hybrid"):
        raise ValueError(f"Unknown search method: {method}")

    output_path = run_search_dataset(
            dataset_path=dataset_path,
            save_directory=save_directory,
            k=k,
            method= method
        )


    console.print(f"[green]Saved to:[/green] {output_path}")


def answer(query: str, k: int = 5):

    if k <= 0:
        raise ValueError("k must be greater than 0")

    bm25 = create_bm25()

    results = bm25.search(query, k)

    if not results:
        console.print("[yellow]No sources found[/yellow]")
        return

    context_list:list[str] = []
    for (number, chunk) in  enumerate(results):
        context_list.append(f"contetn {number + 1}:  {chunk.content}")

    # print(context_list)
    context:str = "\n\n\n".join(context_list)
    print(context)
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

def evaluate(
    student_search_results_path: str,
    dataset_path: str,
):
    return run_evaluate(
        student_search_results_path,
        dataset_path,
    )

