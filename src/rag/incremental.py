import hashlib
import json
from pathlib import Path

from tqdm import tqdm


def file_hash(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def detect_changes(
    directory: str,
    old_hashes: dict[str, str],
) -> tuple[list[Path], set[str], dict[str, str]]:
    new_hashes = {}

    changed_files = []
    for path in tqdm(sorted(Path(directory).rglob("*")),
                     desc="Checking for changes", unit="entry"):
        if not path.is_file():
            continue

        if path.suffix not in (".py", ".md"):
            continue

        key = str(path)
        current_hash = file_hash(path)

        new_hashes[key] = current_hash

        if old_hashes.get(key) != current_hash:
            changed_files.append(path)
    deleted_files = set(old_hashes) - set(new_hashes)

    return changed_files, deleted_files, new_hashes


def load_hashes(
    max_chunk_size: int,
    chunks_mtime_ns: int,
    path: str = "data/processed/file_hashes.json",
) -> dict[str, str] | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, ValueError):
        return None

    if (
        isinstance(data, dict)
        and data.get("max_chunk_size") == max_chunk_size
        and data.get("chunks_mtime_ns") == chunks_mtime_ns
        and isinstance(data.get("hashes"), dict)
    ):
        return {str(key): str(value) for key, value in data["hashes"].items()}
    return None


def save_hashes(
    hashes: dict[str, str],
    max_chunk_size: int,
    chunks_mtime_ns: int,
    path: str = "data/processed/file_hashes.json",
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump({
            "max_chunk_size": max_chunk_size,
            "chunks_mtime_ns": chunks_mtime_ns,
            "hashes": hashes,
        }, file, indent=2)
