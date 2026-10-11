import os
import argparse
from chunking import chunk_file, chunk_file_py, Chunk, store_chunks
import ast
import textwrap

parser = argparse.ArgumentParser()
parser.add_argument("--max_chunk_size")

MAX_CHUNK_SIZE = 2000
VLLM_PATH = "src"


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
            elif en.is_file() and en.name.endswith((".md")):
                with open(en, "r") as file:
                    content = file.read()
                    x = chunk_file(content, en.path, max_chunk_size)
                chunks_list.extend(x)
    return chunks_list


def test():

    content = """
    def add(a, b):
        return a + b

    def test():
        pass
    
    @test
    def multiply(a:int, b):
        return a * b

    """
    content = textwrap.dedent(content)
    tree = ast.parse(content)

    print(ast.dump(tree, indent=4))


def main() -> None:
    args = parser.parse_args()
    max_chunk_size = int(args.max_chunk_size if args.max_chunk_size else MAX_CHUNK_SIZE)
    if max_chunk_size <= 0:
        raise ValueError("max_chunk must be positive")

    try:
        chunks = search_in_vllm_files(VLLM_PATH, max_chunk_size)
        store_chunks(chunks, "data/processed")
        # test()
    except OSError as e:
        print(e)


main()
