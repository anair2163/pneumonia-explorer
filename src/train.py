from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
try:
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from torchvision.models import ResNet18_Weights
    ML_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    Adam = None
    DataLoader = None
    Dataset = None
    models = None
    transforms = None
    ResNet18_Weights = None
    ML_IMPORT_ERROR = exc
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_data import (
    DEFAULT_SPLIT_RANDOM_STATE,
    DEFAULT_VALIDATION_FRACTION,
    build_raw_metadata_dataframe,
    create_effective_split_metadata,
    resolve_dataset_root,
    resolve_project_path,
    validate_dataset_layout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.pt"
HISTORY_JSON_PATH = MODELS_DIR / "baseline_history.json"
HISTORY_CSV_PATH = MODELS_DIR / "baseline_history.csv"
AUG_MODEL_PATH = MODELS_DIR / "aug_model.pt"
AUG_HISTORY_JSON_PATH = MODELS_DIR / "aug_history.json"
AUG_HISTORY_CSV_PATH = MODELS_DIR / "aug_history.csv"
BASELINE_SUMMARY_PATH = MODELS_DIR / "baseline_eval_summary.json"
AUG_SUMMARY_PATH = MODELS_DIR / "aug_eval_summary.json"
EXPERIMENT_CONFIGS = {
    "baseline": {
        "artifact_paths": {
            "model": MODEL_PATH,
            "history_json": HISTORY_JSON_PATH,
            "history_csv": HISTORY_CSV_PATH,
            "summary": BASELINE_SUMMARY_PATH,
        },
        "training_strategy": "Baseline transfer learning with light image augmentation.",
        "augmentation_note": (
            "Baseline uses light training augmentation only. This is the reference run for comparison."
        ),
    },
    "augmented": {
        "artifact_paths": {
            "model": AUG_MODEL_PATH,
            "history_json": AUG_HISTORY_JSON_PATH,
            "history_csv": AUG_HISTORY_CSV_PATH,
            "summary": AUG_SUMMARY_PATH,
        },
        "training_strategy": (
            "Augmentation experiment with stronger geometric and photometric transforms."
        ),
        "augmentation_note": (
            "Augmented uses stronger geometric and photometric augmentation as an augmentation "
            "experiment only, not as truly synthetic clinical image generation."
        ),
    },
}


def require_ml_dependencies() -> None:
    if ML_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Missing dependency: install `torch` and `torchvision` before running training. "
            "The recommended setup command is `pip install -r requirements.txt`."
        ) from ML_IMPORT_ERROR


def set_seed(seed: int = 42) -> None:
    require_ml_dependencies()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    require_ml_dependencies()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(
    image_size: int = 224,
    *,
    experiment_name: Literal["baseline", "augmented"] = "baseline",
) -> dict[str, transforms.Compose]:
    require_ml_dependencies()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if experiment_name == "augmented":
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    shear=4,
                ),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=7),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return {
        "train": train_transform,
        "val": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def build_datasets(
    data_root: Path,
    *,
    image_size: int = 224,
    experiment_name: Literal["baseline", "augmented"] = "baseline",
) -> dict[str, Dataset]:
    require_ml_dependencies()
    transforms_map = build_transforms(image_size=image_size, experiment_name=experiment_name)
    raw_metadata = build_raw_metadata_dataframe(data_root)
    effective_metadata, split_summary = create_effective_split_metadata(
        raw_metadata,
        validation_fraction=DEFAULT_VALIDATION_FRACTION,
        random_state=DEFAULT_SPLIT_RANDOM_STATE,
    )

    return {
        split: MetadataImageDataset(
            effective_metadata[effective_metadata["split"] == split].reset_index(drop=True),
            transform=transforms_map[split],
        )
        for split in ("train", "val", "test")
    } | {"split_summary": split_summary}


class MetadataImageDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, *, transform=None) -> None:
        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform
        self.classes = list(EXPECTED_CLASS_NAMES)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = resolve_project_path(row["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label_index = self.classes.index(row["label"])
        return image, label_index


EXPECTED_CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def build_dataloaders(
    data_root: Path,
    *,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    experiment_name: Literal["baseline", "augmented"] = "baseline",
) -> dict[str, DataLoader]:
    require_ml_dependencies()
    datasets_map = build_datasets(
        data_root,
        image_size=image_size,
        experiment_name=experiment_name,
    )
    split_summary = datasets_map.pop("split_summary")
    return {
        split: DataLoader(
            datasets_map[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in datasets_map
    } | {"split_summary": split_summary}


def build_model(num_classes: int = 2) -> nn.Module:
    require_ml_dependencies()
    weights = None
    try:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("[train] Loaded ResNet18 with pretrained weights.")
    except Exception as exc:
        print(f"[train] Pretrained weights were not available. Falling back to randomly initialized weights. Reason: {exc}")
        model = models.resnet18(weights=None)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    require_ml_dependencies()
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    running_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    epoch_loss = running_loss / max(total_examples, 1)
    epoch_accuracy = running_correct / max(total_examples, 1)
    return epoch_loss, epoch_accuracy


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    require_ml_dependencies()
    with torch.inference_mode():
        loss, accuracy = run_epoch(model, dataloader, criterion, device, optimizer=None)
    return {"loss": loss, "accuracy": accuracy}


def save_training_history(history: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(history, indent=2))
    pd.DataFrame(history).to_csv(csv_path, index=False)


def log_dataset_summary(dataloaders: dict[str, DataLoader]) -> None:
    split_summary = dataloaders["split_summary"]
    train_size = len(dataloaders["train"].dataset)
    val_size = len(dataloaders["val"].dataset)
    test_size = len(dataloaders["test"].dataset)
    total_size = train_size + val_size + test_size

    print("[train] Dataset summary")
    print(f"[train] Total images: {total_size:,}")
    print(f"[train] Original train images: {split_summary['original_train_count']:,}")
    print(f"[train] New train images: {split_summary['new_train_count']:,}")
    print(f"[train] New validation images: {split_summary['new_validation_count']:,}")
    print(f"[train] Original validation images not used for model selection: {split_summary['original_val_count']:,}")
    print(f"[train] Test images (original test split, unchanged): {test_size:,}")


def save_evaluation_summary(summary: dict, summary_path: Path) -> None:
    summary_path.write_text(json.dumps(summary, indent=2))


def get_experiment_paths(experiment_name: Literal["baseline", "augmented"]) -> dict[str, Path]:
    return EXPERIMENT_CONFIGS[experiment_name]["artifact_paths"]


def run_training_experiment(
    *,
    experiment_name: Literal["baseline", "augmented"],
    data_root: Path | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    require_ml_dependencies()
    set_seed(seed)
    layout = validate_dataset_layout(data_root)
    if not layout["is_complete"]:
        missing = "\n".join(layout["missing_folders"])
        raise FileNotFoundError(f"Dataset layout is incomplete. Missing folders:\n{missing}")

    dataset_root = resolve_dataset_root(data_root)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_paths = get_experiment_paths(experiment_name)
    experiment_config = EXPERIMENT_CONFIGS[experiment_name]

    device = get_device()
    print(f"[train] Using dataset root: {dataset_root}")
    print(f"[train] Running {experiment_name} experiment on device: {device}")
    dataloaders = build_dataloaders(
        dataset_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        experiment_name=experiment_name,
    )
    log_dataset_summary(dataloaders)

    class_names = dataloaders["train"].dataset.classes
    if len(class_names) != 2:
        raise ValueError(
            "This training pipeline expects exactly 2 classes. "
            f"Found classes: {class_names}"
        )

    print(
        f"[train] Starting {experiment_name} experiment. "
        f"{experiment_config['training_strategy']}"
    )
    print(
        f"[train] Artifact targets: model={artifact_paths['model'].name}, "
        f"history_json={artifact_paths['history_json'].name}, "
        f"history_csv={artifact_paths['history_csv'].name}, "
        f"summary={artifact_paths['summary'].name}"
    )
    print(
        "[train] Validation policy: the original Kaggle test split remains untouched, "
        "and a new stratified validation split is built from the original training data."
    )
    print(f"[train] Augmentation note: {experiment_config['augmentation_note']}")
    print(f"[train] Class names: {class_names}")
    print(f"[train] Training for {epochs} epoch(s) with batch size {batch_size} and learning rate {learning_rate}.")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    history: list[dict] = []
    best_val_accuracy = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, dataloaders["train"], criterion, device, optimizer=optimizer
        )
        val_metrics = evaluate_model(model, dataloaders["val"], criterion, device)

        epoch_record = {
            "epoch": epoch,
            "experiment": experiment_name,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
        }
        history.append(epoch_record)

        print(
            f"[train] Epoch {epoch}/{epochs} | "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"train_acc={epoch_record['train_accuracy']:.4f} "
            f"val_loss={epoch_record['val_loss']:.4f} "
            f"val_acc={epoch_record['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate_model(model, dataloaders["test"], criterion, device)

    checkpoint = {
        "experiment_name": experiment_name,
        "training_strategy": experiment_config["training_strategy"],
        "augmentation_note": experiment_config["augmentation_note"],
        "model_state_dict": best_state_dict,
        "class_names": class_names,
        "image_size": image_size,
        "best_val_accuracy": round(best_val_accuracy, 6),
        "test_loss": round(test_metrics["loss"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
    }
    torch.save(checkpoint, artifact_paths["model"])

    history_with_test = history + [
        {
            "epoch": "test",
            "experiment": experiment_name,
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": None,
            "val_accuracy": None,
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
        }
    ]
    save_training_history(
        history_with_test,
        artifact_paths["history_json"],
        artifact_paths["history_csv"],
    )

    evaluation_summary = {
        "experiment_name": experiment_name,
        "training_strategy": experiment_config["training_strategy"],
        "augmentation_note": experiment_config["augmentation_note"],
        "class_names": class_names,
        "image_size": image_size,
        "best_val_accuracy": round(best_val_accuracy, 6),
        "test_loss": round(test_metrics["loss"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    save_evaluation_summary(evaluation_summary, artifact_paths["summary"])

    print(f"[train] Saved model checkpoint to: {artifact_paths['model']}")
    print(f"[train] Saved training history JSON to: {artifact_paths['history_json']}")
    print(f"[train] Saved training history CSV to: {artifact_paths['history_csv']}")
    print(f"[train] Saved evaluation summary to: {artifact_paths['summary']}")

    return {
        "experiment_name": experiment_name,
        "model_path": str(artifact_paths["model"]),
        "history_json_path": str(artifact_paths["history_json"]),
        "history_csv_path": str(artifact_paths["history_csv"]),
        "evaluation_summary_path": str(artifact_paths["summary"]),
        "best_val_accuracy": round(best_val_accuracy, 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "class_names": class_names,
    }


def train_baseline_model(
    *,
    data_root: Path | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    return run_training_experiment(
        experiment_name="baseline",
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )


def train_augmented_model(
    *,
    data_root: Path | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
) -> dict:
    return run_training_experiment(
        experiment_name="augmented",
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train baseline or augmented ResNet18 pneumonia classifiers from the chest X-ray dataset.\n\n"
            "Examples:\n"
            "  python src/train.py --epochs 1\n"
            "  python src/train.py --experiment augmented --epochs 1"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "augmented"],
        default="baseline",
        help="Choose whether to run the baseline or stronger augmentation experiment.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--image-size", type=int, default=224, help="Image resize dimension.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = train_baseline_model if args.experiment == "baseline" else train_augmented_model
    print(f"[train] Requested experiment: {args.experiment}")
    try:
        results = trainer(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    except ModuleNotFoundError as exc:
        print(f"[train] {exc}")
        raise SystemExit(1)
    except FileNotFoundError as exc:
        print(f"[train] {exc}")
        raise SystemExit(1)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
