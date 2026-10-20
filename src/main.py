import os
import argparse
from chunking import chunk_file, chunk_file_py, Chunk, store_chunks
from rich.console import Console
from bm25 import BM25Search
from pathlib import Path
from llm import LLM
import json
from search_dataset import search_dataset


parser = argparse.ArgumentParser()


subparser = parser.add_subparsers(dest="command")


# search
search_parser = subparser.add_parser("search")
search_parser.add_argument("query")
search_parser.add_argument("--k", type=int, default=5)

# search_dataset
dataset_parser = subparser.add_parser("search_dataset")
dataset_parser.add_argument("--dataset_path", required=True)
dataset_parser.add_argument("--save_directory", required=True)
dataset_parser.add_argument("--k", type=int, default=5)

# index
index_parser = subparser.add_parser("index")
index_parser.add_argument("--max_chunk_size", type=int, default=2000)

# answer
answer_parser = subparser.add_parser("answer")
answer_parser.add_argument("query")
answer_parser.add_argument("--k", type=int, default=5)

# answer_dataset
dataset_parser = subparser.add_parser("answer_dataset")
dataset_parser.add_argument("--k", type=int, default=5)
dataset_parser.add_argument("--student_search_results_path")
dataset_parser.add_argument("--save_directory")



MAX_CHUNK_SIZE = 2000
VLLM_PATH = "data/raw"

error_console = Console(stderr=True)


def search_in_vllm_files(
    dir: str = VLLM_PATH, max_chunk_size: int = MAX_CHUNK_SIZE
) -> list[Chunk]:
    chunks_list: list[Chunk] = []
    with os.scandir(dir) as entries:
        for en in entries:
            if en.is_dir():
                chunks_list.extend(search_in_vllm_files(en.path, max_chunk_size))
            elif en.is_file() and en.name.endswith((".py")):
                with open(en, "r") as file:
                    content = file.read()
                x = chunk_file_py(content, en.path, max_chunk_size)
                chunks_list.extend(x)
            elif en.is_file() and en.name.endswith((".md")):
                with open(en, "r") as file:
                    content = file.read()
                x = chunk_file(content, en.path, max_chunk_size)
                chunks_list.extend(x)
    return chunks_list


def load_chunks(
    folder_path: str = "data/processed",
) -> list[Chunk]:
    file_path = Path(folder_path) / "chunks.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [Chunk.model_validate(item) for item in data]


def main() -> None:
    args = parser.parse_args()
    console = Console()

    if args.command == "index":
        max_chunk_size = args.max_chunk_size

        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than 0")

        chunks = search_in_vllm_files(VLLM_PATH, max_chunk_size)
        store_chunks(chunks, "data/processed")

        console.print(
            f"[bold green]Indexed {len(chunks)} chunks[/bold green]"
        )

    elif args.command == "search_dataset":
        chunks = load_chunks()

        bm25 = BM25Search()
        bm25.index(chunks)

        output_path = search_dataset(
            dataset_path=args.dataset_path,
            save_directory=args.save_directory,
            bm25=bm25,
            k=args.k,
        )

        print(f"Saved to: {output_path}")

    elif args.command == "search":
        if args.k <= 0:
            raise ValueError("k must be greater than 0")

        chunks = load_chunks()

        bm25 = BM25Search()
        bm25.index(chunks)

        results = bm25.search(args.query, args.k)
        # sources = [
        #     MinimalSource(
        #         file_path=chunk.path,
        #         first_character_index=chunk.start,
        #         last_character_index=chunk.end,
        #     )
        #     for chunk in results
        # ]
        # print("sources",sources)

        # chunks = load_chunks()
        # console.print(
        #     f"[bold yellow]Question:[/bold yellow] {args.query}"
        # )
        # console.print(f"[cyan]Top K:[/cyan] {args.k}")

        # for chunk in results:
        #     console.print(f"\n[green]Path:[/green] {chunk.path}")
        #     console.print(f"Size: {chunk.size}")
        #     console.print(f"Start: {chunk.start}")
        #     console.print(f"End: {chunk.end}")
        #     console.print(f"Content:\n{chunk.content}")

    elif args.command == "answer":
        if args.k <= 0:
            raise ValueError("k must be greater than 0")

        chunks = load_chunks()

        bm25 = BM25Search()
        bm25.index(chunks)

        results = bm25.search(args.query, args.k)

        if not results:
            console.print("[yellow]No sources found[/yellow]")
            return

        context = "\n\n".join(
            chunk.content for chunk in results
        )

        llm = LLM()
        answer = llm.answer(args.query, context)

        console.print(f"[bold green]Answer:[/bold green]")
        print(answer)




try:
    main()
except Exception as e:
    error_console.print(f"[bold red]Error: {e}[/bold red]")
