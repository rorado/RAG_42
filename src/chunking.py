from pathlib import Path
import ast
import json
import uuid

from pydantic import BaseModel, Field


class Chunk(BaseModel):

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

    if max_chunk <= 0:
        raise ValueError("max_chunk must be greater than zero")

    if not content:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = min(start + max_chunk, len(content))

        if end < len(content):
            # Prefer splitting within the last 30% of the chunk
            min_end = start + int(max_chunk * 0.7)

            if min_end < end:
                newline = content.rfind("\n", min_end, end)
                space = content.rfind(" ", min_end, end)

                if newline != -1:
                    end = newline + 1
                elif space != -1:
                    end = space + 1

        chunks.append(
            Chunk(
                content=content[start:end],
                size=end - start,
                path=path,
                start=start,
                end=end,
            )
        )

        start = end

    return chunks


def chunk_file_py(content: str, path: str, max_chunk: int) -> list[Chunk]:
    if max_chunk <= 0:
        raise ValueError("max_chunk must be positive")

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

    boundaries = [0]

    for node in tree.body:
        boundaries.append(offsets[node.lineno - 1])

    boundaries = sorted(set(boundaries + [len(content)]))

    chunks = []
    start = 0

    while start < len(content):
        limit = min(start + max_chunk, len(content))

        possible = [b for b in boundaries if start < b <= limit]

        if possible:
            end = possible[-1]
        else:
            newline = content.rfind("\n", start, limit)
            end = newline + 1 if newline >= start else limit

        chunks.append(
            Chunk(
                content=content[start:end],
                size=end - start,
                path=path,
                start=start,
                end=end,
            )
        )

        start = end

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
