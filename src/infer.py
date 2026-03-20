from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

try:
    import torch
    ML_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    ML_IMPORT_ERROR = exc
from PIL import Image, UnidentifiedImageError

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.train import AUG_MODEL_PATH, MODEL_PATH, build_model, build_transforms, get_device


DEFAULT_MODEL_PATH = MODEL_PATH
MODEL_REGISTRY = {
    "baseline": MODEL_PATH,
    "augmented": AUG_MODEL_PATH,
}


def require_inference_dependencies() -> None:
    if ML_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "torch is required for inference. Install the packages from requirements.txt before using the Prediction Lab."
        ) from ML_IMPORT_ERROR


def load_saved_model(model_path: str | Path | None = None) -> tuple[torch.nn.Module, dict, torch.device]:
    require_inference_dependencies()
    resolved_model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at `{resolved_model_path}`. Train a model before running inference."
        )

    checkpoint = torch.load(resolved_model_path, map_location="cpu")
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = get_device()
    model = model.to(device)
    model.eval()
    return model, checkpoint, device


def preprocess_uploaded_image(uploaded_file, image_size: int) -> torch.Tensor:
    require_inference_dependencies()
    image = _open_image(uploaded_file)
    transform = build_transforms(image_size=image_size, experiment_name="baseline")["test"]
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_image(
    uploaded_file,
    *,
    model_path: str | Path | None = None,
) -> dict:
    require_inference_dependencies()
    model, checkpoint, device = load_saved_model(model_path)
    image_size = int(checkpoint.get("image_size", 224))
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])

    with torch.inference_mode():
        input_tensor = preprocess_uploaded_image(uploaded_file, image_size=image_size).to(device)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_probability = float(probabilities[predicted_index].item())

    return {
        "predicted_class": class_names[predicted_index],
        "predicted_probability": round(predicted_probability, 6),
        "class_probabilities": {
            class_name: round(float(probabilities[idx].item()), 6)
            for idx, class_name in enumerate(class_names)
        },
        "model_path": str(Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH),
        "image_size": image_size,
    }


def predict_baseline_image(uploaded_file) -> dict:
    return predict_image(uploaded_file, model_path=MODEL_PATH)


def predict_placeholder(uploaded_file, threshold: float) -> dict:
    _ = threshold
    result = predict_image(uploaded_file)
    result["threshold"] = threshold
    return result


def predict_from_experiment(uploaded_file, experiment_name: str = "baseline") -> dict:
    if experiment_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown experiment `{experiment_name}`. Expected one of: {', '.join(MODEL_REGISTRY)}."
        )
    return predict_image(uploaded_file, model_path=MODEL_REGISTRY[experiment_name])


def _open_image(uploaded_file) -> Image.Image:
    try:
        if isinstance(uploaded_file, (str, Path)):
            image = Image.open(uploaded_file)
        elif isinstance(uploaded_file, bytes):
            image = Image.open(BytesIO(uploaded_file))
        elif hasattr(uploaded_file, "read"):
            raw_bytes = uploaded_file.read()
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            image = Image.open(BytesIO(raw_bytes))
        else:
            raise ValueError("Unsupported image input type for inference.")

        return image.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find the uploaded image: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "The uploaded file could not be opened as a valid image. Please use a PNG or JPEG chest X-ray image."
        ) from exc
