from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
try:
    import torch
    ML_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    torch = None
    ML_IMPORT_ERROR = exc
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.infer import MODEL_REGISTRY, _open_image, load_saved_model, predict_image, preprocess_uploaded_image


EXPLANATION_METHOD = "grad_cam_lightweight"


def require_explain_dependencies() -> None:
    if ML_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "torch is required for explainability features. Install the packages from requirements.txt before requesting Grad-CAM output."
        ) from ML_IMPORT_ERROR


def explain_prediction(
    uploaded_file,
    *,
    experiment_name: str = "baseline",
    model_path: str | Path | None = None,
    alpha: float = 0.4,
) -> dict:
    require_explain_dependencies()
    resolved_model_path = _resolve_model_path(experiment_name, model_path)
    model, checkpoint, device = load_saved_model(resolved_model_path)
    image_size = int(checkpoint.get("image_size", 224))
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])

    original_image = _open_image(uploaded_file)
    input_tensor = preprocess_uploaded_image(uploaded_file, image_size=image_size).to(device)

    target_layer = _get_target_layer(model)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probabilities, dim=1).item())
        target_score = logits[:, predicted_index]
        target_score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if not activations or not gradients:
        raise RuntimeError(
            "Grad-CAM activations were not captured. This helper assumes a ResNet-style model with an accessible final convolution block."
        )

    heatmap = _build_gradcam_heatmap(activations[-1], gradients[-1])
    heatmap_image = _heatmap_to_image(heatmap, original_image.size)
    overlay_image = _overlay_heatmap_on_image(original_image, heatmap_image, alpha=alpha)

    prediction = predict_image(uploaded_file, model_path=resolved_model_path)
    prediction.update(
        {
            "status": "available",
            "explanation_available": True,
            "explanation_method": EXPLANATION_METHOD,
            "explanation_note": (
                "This is a lightweight Grad-CAM style overlay. Brighter regions indicate areas that contributed more strongly "
                "to the model's current prediction, but the view is illustrative rather than clinical."
            ),
            "original_image": original_image,
            "heatmap_image": heatmap_image,
            "overlay_image": overlay_image,
            "predicted_class": class_names[predicted_index],
            "predicted_probability": round(float(probabilities[0, predicted_index].item()), 6),
            "overlay_alpha": round(max(0.0, min(alpha, 1.0)), 2),
        }
    )
    return prediction


def try_explain_prediction(
    uploaded_file,
    *,
    experiment_name: str = "baseline",
    model_path: str | Path | None = None,
    alpha: float = 0.4,
) -> dict:
    """Return a stable payload for UI usage, even when explanation generation fails.

    This keeps the first-pass Streamlit integration simple and gives the page a
    clean place to fall back to a polished status card. If the architecture
    changes later, this function is also a natural place to swap in a more
    complete Grad-CAM implementation.
    """
    try:
        return explain_prediction(
            uploaded_file,
            experiment_name=experiment_name,
            model_path=model_path,
            alpha=alpha,
        )
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError, AttributeError) as exc:
        return {
            "status": "unavailable",
            "explanation_available": False,
            "explanation_method": EXPLANATION_METHOD,
            "explanation_note": (
                "A Grad-CAM style overlay is planned here, but it is not available for the current "
                "model, environment, or input image."
            ),
            "reason": str(exc),
        }


def get_explanation_image(
    uploaded_file,
    *,
    experiment_name: str = "baseline",
    model_path: str | Path | None = None,
    alpha: float = 0.4,
) -> Image.Image:
    explanation = explain_prediction(
        uploaded_file,
        experiment_name=experiment_name,
        model_path=model_path,
        alpha=alpha,
    )
    return explanation["overlay_image"]


def get_augmentation_notes() -> str:
    return (
        "This section treats stronger image transformations as an augmentation experiment, not as evidence of truly synthetic clinical image generation. "
        "The explainability helper is a lightweight Grad-CAM style overlay intended for demo and interpretation support."
    )


def _resolve_model_path(experiment_name: str, model_path: str | Path | None) -> Path:
    if model_path is not None:
        return Path(model_path)
    if experiment_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown experiment `{experiment_name}`. Expected one of: {', '.join(MODEL_REGISTRY)}."
        )
    return MODEL_REGISTRY[experiment_name]


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # For the current ResNet18 baseline, the final convolution block is a sensible first-pass target.
    # If the architecture changes later, this function is the clean place to swap in a different layer.
    return model.layer4[-1].conv2


def _build_gradcam_heatmap(
    activations: torch.Tensor,
    gradients: torch.Tensor,
) -> np.ndarray:
    pooled_gradients = gradients.mean(dim=(2, 3), keepdim=True)
    weighted_activations = pooled_gradients * activations
    cam = weighted_activations.sum(dim=1).squeeze(0)
    cam = torch.relu(cam)

    cam_max = cam.max()
    if torch.isclose(cam_max, torch.tensor(0.0, device=cam.device)):
        return np.zeros(cam.shape, dtype=np.float32)

    normalized = cam / cam_max
    return normalized.detach().cpu().numpy().astype(np.float32)


def _heatmap_to_image(heatmap: np.ndarray, image_size: tuple[int, int]) -> Image.Image:
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    base = Image.fromarray(heatmap_uint8, mode="L").resize(image_size, Image.Resampling.BILINEAR)

    # Simple red-yellow palette for a readable first-pass overlay.
    heatmap_array = np.asarray(base).astype(np.float32) / 255.0
    red = np.clip(255 * heatmap_array, 0, 255)
    green = np.clip(180 * heatmap_array, 0, 255)
    blue = np.clip(70 * (1 - heatmap_array), 0, 255)
    colorized = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(colorized, mode="RGB")


def _overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap_image: Image.Image,
    *,
    alpha: float,
) -> Image.Image:
    original_rgb = original_image.convert("RGB")
    resized_heatmap = heatmap_image.resize(original_rgb.size, Image.Resampling.BILINEAR)
    return Image.blend(original_rgb, resized_heatmap, alpha=max(0.0, min(alpha, 1.0)))
