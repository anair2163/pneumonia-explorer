from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_LABELS = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}
DEFAULT_VALIDATION_FRACTION = 0.12
DEFAULT_SPLIT_RANDOM_STATE = 42


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_root() -> Path:
    return DATA_RAW_DIR


def to_project_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def detect_dataset_root(
    data_root: Path | None = None,
    *,
    verbose: bool = True,
    require_complete: bool = False,
) -> Path:
    base_root = Path(data_root) if data_root is not None else DATA_RAW_DIR

    if not base_root.exists():
        raise FileNotFoundError(
            "Dataset directory was not found. Expected raw data under "
            f"`{base_root}`. Run `python scripts/download_data.py` first."
        )

    candidate_roots = _build_dataset_root_candidates(base_root)
    inspected: list[tuple[Path, list[str], bool]] = []

    for candidate in candidate_roots:
        present_splits = [split for split in EXPECTED_SPLITS if (candidate / split).exists()]
        if not present_splits:
            continue

        is_complete = _is_complete_dataset_root(candidate)
        inspected.append((candidate, present_splits, is_complete))
        if require_complete and not is_complete:
            continue

        if verbose:
            print(
                "[load_data] Selected dataset root:",
                to_project_relative_path(candidate),
                f"(splits: {', '.join(present_splits)}; complete={is_complete})",
            )
        return candidate

    if verbose:
        if inspected:
            print("[load_data] Candidate dataset roots were found but none matched the requested completeness.")
            for candidate, present_splits, is_complete in inspected:
                print(
                    "[load_data] Candidate:",
                    to_project_relative_path(candidate),
                    f"(splits: {', '.join(present_splits)}; complete={is_complete})",
                )
        else:
            print(
                "[load_data] No dataset root matched expected layouts under:",
                to_project_relative_path(base_root),
            )

    if not inspected:
        raise FileNotFoundError(
            "Dataset directory does not contain the expected split folders "
            f"{EXPECTED_SPLITS}. Checked `{base_root}` and common nested alternatives."
        )

    raise FileNotFoundError(
        "Dataset root candidates were found, but none contained the full expected "
        f"{EXPECTED_SPLITS} and {EXPECTED_LABELS} layout."
    )


def resolve_dataset_root(data_root: Path | None = None) -> Path:
    return detect_dataset_root(data_root, verbose=True, require_complete=False)


def validate_dataset_layout(data_root: Path | None = None) -> dict:
    dataset_root = detect_dataset_root(data_root, verbose=True, require_complete=False)
    missing_folders: list[str] = []

    for split in EXPECTED_SPLITS:
        for label in EXPECTED_LABELS:
            expected_dir = dataset_root / split / label
            if not expected_dir.exists():
                missing_folders.append(to_project_relative_path(expected_dir))

    return {
        "dataset_root": dataset_root,
        "is_complete": not missing_folders,
        "missing_folders": missing_folders,
    }


def build_metadata_dataframe(data_root: Path | None = None) -> pd.DataFrame:
    raw_metadata = build_raw_metadata_dataframe(data_root)
    effective_metadata, _ = create_effective_split_metadata(raw_metadata)
    return effective_metadata


def build_raw_metadata_dataframe(data_root: Path | None = None) -> pd.DataFrame:
    layout = validate_dataset_layout(data_root)
    dataset_root = layout["dataset_root"]

    if not layout["is_complete"]:
        missing = "\n".join(layout["missing_folders"])
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing one or more expected class folders:\n"
            f"{missing}"
        )

    records: list[dict[str, str]] = []
    for split in EXPECTED_SPLITS:
        for label in EXPECTED_LABELS:
            class_dir = dataset_root / split / label
            for image_path in sorted(_iter_image_files(class_dir)):
                records.append(
                    {
                        "image_path": to_project_relative_path(image_path),
                        "split": split,
                        "original_split": split,
                        "label": label,
                        "filename": image_path.name,
                    }
                )

    if not records:
        raise FileNotFoundError(
            "No image files were found in the dataset folders. "
            f"Checked `{to_project_relative_path(dataset_root)}` for extensions {sorted(IMAGE_EXTENSIONS)}."
        )

    return pd.DataFrame(records)


def create_effective_split_metadata(
    raw_metadata: pd.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    if raw_metadata.empty:
        raise ValueError("Cannot create an effective split because the raw metadata DataFrame is empty.")

    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    if "original_split" not in raw_metadata.columns:
        raw_metadata = raw_metadata.copy()
        raw_metadata["original_split"] = raw_metadata["split"]

    original_train_df = raw_metadata[raw_metadata["original_split"] == "train"].copy()
    original_val_df = raw_metadata[raw_metadata["original_split"] == "val"].copy()
    original_test_df = raw_metadata[raw_metadata["original_split"] == "test"].copy()

    if original_train_df.empty:
        raise ValueError("Cannot create an effective split because the original training folder is empty.")
    if original_test_df.empty:
        raise ValueError("Cannot create an effective split because the original test folder is empty.")

    train_indices, val_indices = train_test_split(
        original_train_df.index.to_numpy(),
        test_size=validation_fraction,
        stratify=original_train_df["label"],
        random_state=random_state,
    )

    effective_train_df = original_train_df.loc[train_indices].copy()
    effective_val_df = original_train_df.loc[val_indices].copy()
    effective_test_df = original_test_df.copy()

    effective_train_df["split"] = "train"
    effective_val_df["split"] = "val"
    effective_test_df["split"] = "test"

    effective_metadata = pd.concat(
        [effective_train_df, effective_val_df, effective_test_df],
        ignore_index=True,
    )
    effective_metadata = effective_metadata.sort_values(["split", "label", "filename"]).reset_index(drop=True)

    split_summary = {
        "validation_fraction": validation_fraction,
        "random_state": random_state,
        "original_train_count": int(len(original_train_df)),
        "original_val_count": int(len(original_val_df)),
        "original_test_count": int(len(original_test_df)),
        "new_train_count": int(len(effective_train_df)),
        "new_validation_count": int(len(effective_val_df)),
        "test_count": int(len(effective_test_df)),
    }
    return effective_metadata, split_summary


def get_effective_split_summary(
    data_root: Path | None = None,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_SPLIT_RANDOM_STATE,
) -> dict[str, int | float]:
    raw_metadata = build_raw_metadata_dataframe(data_root)
    _, split_summary = create_effective_split_metadata(
        raw_metadata,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    return split_summary


def sample_images(
    metadata: pd.DataFrame,
    *,
    split: str | None = None,
    label: str | None = None,
    n: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    if metadata.empty:
        raise ValueError("Cannot sample images because the metadata DataFrame is empty.")

    filtered = metadata.copy()
    if split is not None:
        _ensure_known_value(split, EXPECTED_SPLITS, "split")
        filtered = filtered[filtered["split"] == split]
    if label is not None:
        _ensure_known_value(label, EXPECTED_LABELS, "label")
        filtered = filtered[filtered["label"] == label]

    if filtered.empty:
        raise ValueError(
            f"No rows matched the requested sample filters: split={split!r}, label={label!r}."
        )

    sample_size = min(n, len(filtered))
    return filtered.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def sample_images_by_group(
    metadata: pd.DataFrame,
    *,
    n_per_group: int = 3,
    splits: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if metadata.empty:
        raise ValueError("Cannot sample images because the metadata DataFrame is empty.")

    selected_splits = tuple(splits) if splits is not None else EXPECTED_SPLITS
    selected_labels = tuple(labels) if labels is not None else EXPECTED_LABELS

    for split in selected_splits:
        _ensure_known_value(split, EXPECTED_SPLITS, "split")
    for label in selected_labels:
        _ensure_known_value(label, EXPECTED_LABELS, "label")

    frames: list[pd.DataFrame] = []
    for split in selected_splits:
        for label in selected_labels:
            group = metadata[(metadata["split"] == split) & (metadata["label"] == label)]
            if group.empty:
                continue
            sample_size = min(n_per_group, len(group))
            frames.append(group.sample(n=sample_size, random_state=random_state))

    if not frames:
        raise ValueError("No groups contained data for the requested split and label filters.")

    return pd.concat(frames, ignore_index=True)


def get_selectable_example_images(
    data_root: Path | None = None,
    *,
    preferred_split: str = "test",
    labels: Iterable[str] = ("NORMAL", "PNEUMONIA"),
    n_per_label: int = 2,
    fallback_splits: Iterable[str] = ("val", "train"),
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a small set of example images for UI galleries and sample selection.

    The helper prefers the requested split first, then gracefully falls back to
    later splits if a preferred group is not available.
    """
    _ensure_known_value(preferred_split, EXPECTED_SPLITS, "preferred_split")

    selected_labels = tuple(labels)
    for label in selected_labels:
        _ensure_known_value(label, EXPECTED_LABELS, "label")

    fallback_order = [preferred_split]
    for split in fallback_splits:
        _ensure_known_value(split, EXPECTED_SPLITS, "fallback split")
        if split not in fallback_order:
            fallback_order.append(split)

    metadata = build_metadata_dataframe(data_root)
    frames: list[pd.DataFrame] = []

    for label in selected_labels:
        selected_group: pd.DataFrame | None = None
        selected_split: str | None = None

        for split in fallback_order:
            group = metadata[(metadata["split"] == split) & (metadata["label"] == label)]
            if not group.empty:
                selected_group = group
                selected_split = split
                break

        if selected_group is None or selected_split is None:
            continue

        sample_size = min(n_per_label, len(selected_group))
        sampled = selected_group.sample(n=sample_size, random_state=random_state).copy()
        sampled["source_split"] = selected_split
        frames.append(sampled)

    if not frames:
        raise ValueError(
            "No example images were available from the dataset for the requested sample groups."
        )

    return pd.concat(frames, ignore_index=True)


def inspect_image_size_distribution(
    metadata: pd.DataFrame,
    *,
    max_images: int | None = 500,
) -> pd.DataFrame:
    if metadata.empty:
        raise ValueError("Cannot inspect image sizes because the metadata DataFrame is empty.")

    subset = metadata.head(max_images) if max_images is not None else metadata
    records: list[dict[str, object]] = []
    unreadable_files: list[str] = []

    for row in subset.itertuples(index=False):
        image_path = resolve_project_path(row.image_path)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            unreadable_files.append(str(image_path))
            continue

        records.append(
            {
                "image_path": to_project_relative_path(image_path),
                "split": row.split,
                "label": row.label,
                "filename": row.filename,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4) if height else None,
            }
        )

    if not records:
        detail = unreadable_files[0] if unreadable_files else "No readable files found."
        raise ValueError(f"Could not read any images to inspect size distribution. Example: {detail}")

    size_df = pd.DataFrame(records)
    size_df.attrs["unreadable_files"] = unreadable_files
    return size_df


def get_dataset_overview(data_root: Path | None = None) -> dict:
    metadata = build_metadata_dataframe(data_root)
    class_counts = (
        metadata.groupby(["split", "label"]).size().reset_index(name="count").sort_values(["split", "label"])
    )
    split_summary = get_effective_split_summary(data_root)

    size_mode = "Unavailable"
    try:
        size_df = inspect_image_size_distribution(metadata, max_images=250)
        if not size_df.empty:
            top_size = (
                size_df.groupby(["width", "height"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .iloc[0]
            )
            size_mode = f"{int(top_size['width'])} x {int(top_size['height'])}"
    except ValueError:
        pass

    schema = pd.DataFrame(
        [
            {"column": "image_path", "description": "Project-relative path to the chest X-ray image file"},
            {"column": "split", "description": "Effective dataset partition: train, val, or test"},
            {"column": "original_split", "description": "Original Kaggle folder split before the effective train/val rebuild"},
            {"column": "label", "description": "Class label: NORMAL or PNEUMONIA"},
            {"column": "filename", "description": "Original image filename"},
        ]
    )

    return {
        "rows": f"{len(metadata):,}",
        "classes": str(metadata["label"].nunique()),
        "image_size": size_mode,
        "schema": schema,
        "class_counts": class_counts,
        "split_summary": split_summary,
        "dataset_root": to_project_relative_path(resolve_dataset_root(data_root)),
    }


def _iter_image_files(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_dataset_root_candidates(base_root: Path) -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        normalized = path.resolve(strict=False)
        if normalized not in candidates and normalized.exists():
            candidates.append(normalized)

    add_candidate(base_root)
    add_candidate(base_root / "chest_xray")

    for child in sorted(path for path in base_root.iterdir() if path.is_dir()):
        add_candidate(child)
        add_candidate(child / "chest_xray")

    return candidates


def _is_complete_dataset_root(candidate: Path) -> bool:
    for split in EXPECTED_SPLITS:
        for label in EXPECTED_LABELS:
            if not (candidate / split / label).exists():
                return False
    return True


def _ensure_known_value(value: str, choices: Iterable[str], field_name: str) -> None:
    if value not in choices:
        raise ValueError(
            f"Unknown {field_name} `{value}`. Expected one of: {', '.join(choices)}."
        )
