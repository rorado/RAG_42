import os
import argparse
from chunking import chunk_file, Chunk

parser = argparse.ArgumentParser()
parser.add_argument("--max_chunk_size")
def search_in_vllm_files(dir: str="./vllm-0.10.1", max_chunk_size: int = 2000):

    with os.scandir(dir) as entries:
        for en in entries:
            if en.is_dir():
                search_in_vllm_files(en)
            elif en.is_file() and en.name.endswith((".txt")):
                with open(en, "r") as file:
                    content = file.read()
                print(chunk_file(content, en.path, max_chunk_size)[0].text)


def main() -> None:
    args = parser.parse_args()
    max_chunk_size = args.max_chunk_size if args.max_chunk_size else 2000
    try:
        search_in_vllm_files("./src", max_chunk_size)
    except OSError as e:
        print(e)

main()
