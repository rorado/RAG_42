import json
from pathlib import Path

from .chunking import Chunk


def load_chunks(folder_path: str = "data/processed") -> list[Chunk]:
    file_path = Path(folder_path) / "chunks.json"

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [Chunk.model_validate(item) for item in data]