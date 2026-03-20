from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ML_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    DataLoader = None
    datasets = None
    ML_IMPORT_ERROR = exc
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.load_data import resolve_dataset_root, to_project_relative_path, validate_dataset_layout
from src.train import (
    AUG_MODEL_PATH,
    AUG_SUMMARY_PATH,
    BASELINE_SUMMARY_PATH,
    MODEL_PATH,
    build_model,
    build_transforms,
    get_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
COMPARISON_JSON_PATH = MODELS_DIR / "evaluation_comparison.json"
COMPARISON_CSV_PATH = MODELS_DIR / "evaluation_comparison.csv"


MODEL_REGISTRY = {
    "baseline": {
        "model_path": MODEL_PATH,
        "summary_path": BASELINE_SUMMARY_PATH,
        "predictions_path": MODELS_DIR / "baseline_predictions.csv",
        "sample_predictions_path": MODELS_DIR / "baseline_sample_predictions.csv",
        "roc_curve_path": MODELS_DIR / "baseline_roc_curve.csv",
    },
    "augmented": {
        "model_path": AUG_MODEL_PATH,
        "summary_path": AUG_SUMMARY_PATH,
        "predictions_path": MODELS_DIR / "aug_predictions.csv",
        "sample_predictions_path": MODELS_DIR / "aug_sample_predictions.csv",
        "roc_curve_path": MODELS_DIR / "aug_roc_curve.csv",
    },
}


def require_evaluation_dependencies() -> None:
    if ML_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "torch and torchvision are required to run model evaluation. "
            "Install the packages from requirements.txt before running `python src/evaluate.py`."
        ) from ML_IMPORT_ERROR


def build_test_dataloader(
    data_root: Path,
    *,
    image_size: int,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, datasets.ImageFolder]:
    require_evaluation_dependencies()
    transform = build_transforms(image_size=image_size, experiment_name="baseline")["test"]
    test_dataset = datasets.ImageFolder(data_root / "test", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: datasets.ImageFolder,
    device: torch.device,
) -> pd.DataFrame:
    require_evaluation_dependencies()
    rows: list[dict] = []
    offset = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)

            for idx_in_batch in range(len(labels)):
                sample_path, _ = dataset.samples[offset + idx_in_batch]
                rows.append(
                    {
                        "image_path": to_project_relative_path(Path(sample_path)),
                        "filename": Path(sample_path).name,
                        "true_label_index": int(labels[idx_in_batch].item()),
                        "predicted_label_index": int(predictions[idx_in_batch].item()),
                        "prob_normal": float(probabilities[idx_in_batch][0].item()),
                        "prob_pneumonia": float(probabilities[idx_in_batch][1].item()),
                    }
                )
            offset += len(labels)

    predictions_df = pd.DataFrame(rows)
    predictions_df["true_label"] = predictions_df["true_label_index"].map(
        {idx: name for idx, name in enumerate(dataset.classes)}
    )
    predictions_df["predicted_label"] = predictions_df["predicted_label_index"].map(
        {idx: name for idx, name in enumerate(dataset.classes)}
    )
    predictions_df["correct"] = (
        predictions_df["true_label_index"] == predictions_df["predicted_label_index"]
    )
    return predictions_df


def compute_metrics(predictions_df: pd.DataFrame) -> tuple[dict, pd.DataFrame | None]:
    y_true = predictions_df["true_label_index"]
    y_pred = predictions_df["predicted_label_index"]
    y_score = predictions_df["prob_pneumonia"]

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "labels": ["NORMAL", "PNEUMONIA"],
        "values": matrix.tolist(),
    }

    roc_curve_df = None
    roc_auc_value = None
    if y_true.nunique() > 1:
        roc_auc_value = round(float(roc_auc_score(y_true, y_score)), 6)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_curve_df = pd.DataFrame(
            {
                "false_positive_rate": fpr,
                "true_positive_rate": tpr,
                "threshold": thresholds,
            }
        )
    metrics["roc_auc"] = roc_auc_value

    return metrics, roc_curve_df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def evaluate_checkpoint(
    experiment_name: str,
    *,
    data_root: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    registry_entry = MODEL_REGISTRY[experiment_name]
    model_path = registry_entry["model_path"]
    if not model_path.exists():
        return {
            "experiment_name": experiment_name,
            "status": "missing_model",
            "message": f"Model file not found at `{model_path}`.",
        }

    require_evaluation_dependencies()

    layout = validate_dataset_layout(data_root)
    if not layout["is_complete"]:
        missing = "\n".join(layout["missing_folders"])
        raise FileNotFoundError(f"Dataset layout is incomplete. Missing folders:\n{missing}")

    dataset_root = resolve_dataset_root(data_root)
    checkpoint = torch.load(model_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", 224))
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])

    device = get_device()
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_loader, test_dataset = build_test_dataloader(
        dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    predictions_df = collect_predictions(model, test_loader, test_dataset, device)
    metrics, roc_curve_df = compute_metrics(predictions_df)

    sample_predictions_df = (
        predictions_df[
            [
                "filename",
                "true_label",
                "predicted_label",
                "prob_normal",
                "prob_pneumonia",
                "correct",
            ]
        ]
        .head(12)
        .copy()
    )

    save_dataframe(predictions_df, registry_entry["predictions_path"])
    save_dataframe(sample_predictions_df, registry_entry["sample_predictions_path"])

    if roc_curve_df is not None:
        save_dataframe(roc_curve_df, registry_entry["roc_curve_path"])

    summary = {
        "experiment_name": experiment_name,
        "status": "evaluated",
        "model_path": to_project_relative_path(model_path),
        "predictions_path": to_project_relative_path(registry_entry["predictions_path"]),
        "sample_predictions_path": to_project_relative_path(registry_entry["sample_predictions_path"]),
        "roc_curve_path": (
            to_project_relative_path(registry_entry["roc_curve_path"]) if roc_curve_df is not None else None
        ),
        "num_test_images": int(len(predictions_df)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    registry_entry["summary_path"].write_text(json.dumps(summary, indent=2))
    print(f"Evaluated {experiment_name} model.")
    print(f"Saved summary to: {registry_entry['summary_path']}")
    print(f"Saved predictions to: {registry_entry['predictions_path']}")
    print(f"Saved sample predictions to: {registry_entry['sample_predictions_path']}")
    if roc_curve_df is not None:
        print(f"Saved ROC curve points to: {registry_entry['roc_curve_path']}")

    return summary


def evaluate_available_models(
    *,
    data_root: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> list[dict]:
    results = []
    for experiment_name in ("baseline", "augmented"):
        print(f"[evaluate] Checking {experiment_name} model...")
        results.append(
            evaluate_checkpoint(
                experiment_name,
                data_root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )

    evaluated_rows = [
        {
            "model": result["experiment_name"].title(),
            "accuracy": result.get("accuracy"),
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1": result.get("f1"),
            "roc_auc": result.get("roc_auc"),
            "status": result["status"],
        }
        for result in results
    ]
    comparison_df = pd.DataFrame(evaluated_rows)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    COMPARISON_JSON_PATH.write_text(json.dumps(results, indent=2))
    return results


def get_evaluation_summary() -> dict:
    available_summaries: list[dict] = []
    for experiment_name in ("baseline", "augmented"):
        summary_path = MODEL_REGISTRY[experiment_name]["summary_path"]
        if summary_path.exists():
            available_summaries.append(json.loads(summary_path.read_text()))

    if not available_summaries:
        comparison = pd.DataFrame(
            [
                {
                    "model": "Baseline",
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                    "status": "not_evaluated",
                },
                {
                    "model": "Augmented",
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "roc_auc": None,
                    "status": "not_evaluated",
                },
            ]
        )
        return {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1": "N/A",
            "comparison": comparison,
        }

    comparison = pd.DataFrame(
        [
            {
                "model": summary["experiment_name"].title(),
                "accuracy": summary.get("accuracy"),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": summary.get("f1"),
                "roc_auc": summary.get("roc_auc"),
                "status": summary.get("status"),
            }
            for summary in available_summaries
        ]
    )
    best_row = comparison.sort_values("f1", ascending=False, na_position="last").iloc[0]

    return {
        "accuracy": _format_metric(best_row.get("accuracy")),
        "precision": _format_metric(best_row.get("precision")),
        "recall": _format_metric(best_row.get("recall")),
        "f1": _format_metric(best_row.get("f1")),
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate available pneumonia classification checkpoints and save app-readable artifacts."
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        results = evaluate_available_models(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except ModuleNotFoundError as exc:
        print(
            "[evaluate] Missing dependency: install `torch` and `torchvision` before running evaluation. "
            "The recommended setup command is `pip install -r requirements.txt`."
        )
        print(f"[evaluate] Details: {exc}")
        raise SystemExit(1)
    except FileNotFoundError as exc:
        print(f"[evaluate] {exc}")
        raise SystemExit(1)

    missing_models = [result for result in results if result.get("status") == "missing_model"]
    if len(missing_models) == len(results):
        print(
            "[evaluate] No model checkpoints were found. "
            "Run `python src/train.py --epochs 1` to create a baseline model, then rerun evaluation."
        )
    print(json.dumps(results, indent=2))


def _format_metric(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
