from __future__ import annotations

import shutil
import sys
from pathlib import Path

import kagglehub

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_data import detect_dataset_root, to_project_relative_path


DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def _is_effectively_downloaded(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    try:
        selected_root = detect_dataset_root(target_dir, verbose=True, require_complete=True)
        print(f"[download] Existing normalized dataset root detected: {to_project_relative_path(selected_root)}")
        return True
    except FileNotFoundError:
        return False


def _sync_directory(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def main() -> None:
    print("[1/4] Ensuring data/raw/ exists...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _is_effectively_downloaded(RAW_DATA_DIR):
        print("[2/4] Existing dataset detected in data/raw/.")
        print(f"[done] Reusing local dataset at: {to_project_relative_path(RAW_DATA_DIR)}")
        return

    print("[2/4] Downloading dataset via kagglehub...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print(f"[3/4] KaggleHub dataset available at: {downloaded_path}")

    try:
        source_dir = detect_dataset_root(downloaded_path, verbose=True, require_complete=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Downloaded dataset did not contain a supported train/val/test layout. "
            f"Inspect this path: {downloaded_path}"
        )

    print(f"[download] Selected KaggleHub dataset root: {to_project_relative_path(source_dir)}")
    print("[4/4] Syncing dataset contents into data/raw/...")
    _sync_directory(source_dir, RAW_DATA_DIR)
    normalized_root = detect_dataset_root(RAW_DATA_DIR, verbose=True, require_complete=True)
    print(f"[done] Dataset ready at: {to_project_relative_path(normalized_root)}")


if __name__ == "__main__":
    main()
