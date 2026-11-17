import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def detect_changes(
    directory: str = "data/raw/vllm-0.10.1",
) -> tuple[list[Path], set[str], dict[str, str]]:
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


def load_hashes(
    path: str = "data/processed/file_hashes.json",
) -> dict[str, str]:
    file_path = Path(path)

    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
            if not isinstance(data, dict):
                return {}
            return {str(key): str(value) for key, value in data.items()}
        except Exception:
            print("chunks not found")
            return {}


def save_hashes(
    hashes: dict[str, str],
    path: str = "data/processed/file_hashes.json",
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(hashes, file, indent=2)
