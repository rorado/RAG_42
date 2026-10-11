from pathlib import Path
import ast
import json
import uuid

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represent a chunk of source text."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    size: int
    path: str
    start: int
    end: int


def chunk_file(
    content: str,
    path: str,
    max_chunk: int,
) -> list[Chunk]:
    """Split text into fixed-size chunks."""
    if max_chunk <= 0:
        raise ValueError("max_chunk must be greater than zero")

    chunks: list[Chunk] = []

    for start in range(0, len(content), max_chunk):
        end = min(start + max_chunk, len(content))
        chunks.append(
            Chunk(
                content=content[start:end],
                size=end - start,
                path=path,
                start=start,
                end=end,
            )
        )

    return chunks



def chunk_file_py(
    content: str,
    path: str,
    max_chunk: int,
) -> list[Chunk]:
    """Split Python source into AST-aware chunks.

    Prefer top-level statement boundaries while preserving
    exact character offsets and the original source text.
    """
    if max_chunk <= 0:
        raise ValueError("max_chunk must be greater than zero")

    if not content:
        return []

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return chunk_file(content, path, max_chunk)

    lines = content.splitlines(keepends=True)
    offsets = [0]

    for line in lines:
        offsets.append(offsets[-1] + len(line))

    chunks: list[Chunk] = []

    def add_chunk(start: int, end: int) -> None:
        if start >= end:
            return

        chunks.append(
            Chunk(
                content=content[start:end],
                size=end - start,
                path=path,
                start=start,
                end=end,
            )
        )

    def split_large(start: int, end: int) -> None:
        """Split oversized blocks, preferring line boundaries."""
        while end - start > max_chunk:
            limit = start + max_chunk

            boundary = content.rfind("\n", start, limit)

            if boundary >= start:
                boundary += 1
            else:
                boundary = limit

            add_chunk(start, boundary)
            start = boundary

        add_chunk(start, end)

    boundaries: list[int] = [0]
    print(path)
    for node in tree.body:
        if not hasattr(node, "lineno"):
            print("path", path)
            continue

        start = offsets[node.lineno - 1]

        boundaries.append(start)
    boundaries.append(len(content))

    chunk_start = 0

    for i in range(1, len(boundaries)):
        block_start = boundaries[i - 1]
        block_end = boundaries[i]

        if block_end - chunk_start <= max_chunk:
            continue

        # Flush previous complete statements.
        if block_start > chunk_start:
            add_chunk(chunk_start, block_start)
            chunk_start = block_start

        # Oversized statement: split it separately.
        if block_end - chunk_start > max_chunk:
            split_large(chunk_start, block_end)
            chunk_start = block_end

    # Save remaining statements.
    if chunk_start < len(content):
        split_large(chunk_start, len(content))

    return chunks


def store_chunks(
    chunks: list[Chunk],
    folder_path: str = "data/processed",
) -> None:
    """Write chunks to a JSON file."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    file_path = folder / "chunks.json"
    serialized_chunks = [chunk.model_dump() for chunk in chunks]

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(serialized_chunks, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Small example you can run to test the chunking functions.
    example = '''import os

def hello():
    print("Hello")

def calculate():
    return 10 + 20
'''

    max_chunk = 50
    result = chunk_file_py(example, "example.py", max_chunk)

    for chunk in result:
        print(f"\nChunk {chunk.start}:{chunk.end} (size={chunk.size})")
        print(chunk.content)

    # Basic correctness checks.
    assert all(chunk.size <= max_chunk for chunk in result)
    assert all(chunk.size == len(chunk.content) for chunk in result)
    assert "".join(chunk.content for chunk in result) == example
