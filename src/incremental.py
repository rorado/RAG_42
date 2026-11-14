
import hashlib
from pathlib import Path
import json
from typing import Any


def file_hash(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()

def detect_changes(directory="data/raw/vllm-0.10.1") -> tuple[list, set, dict]:
    old_hashes = load_hashes()

    new_hashes = {}

    changed_files = []
    for path in sorted(Path(directory).rglob("*")):
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


def load_hashes(path="data/processed/file_hashes.json") -> (dict | Any):
    path = Path(path)

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except Exception:
            print("chunks not found")
            return {}



def save_hashes(hashes, path="data/processed/file_hashes.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=2)
