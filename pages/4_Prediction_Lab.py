from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.explain import try_explain_prediction
from src.infer import MODEL_REGISTRY, predict_image
from src.layout import render_page_header, render_sidebar, render_surface_card, setup_page
from src.load_data import get_selectable_example_images, resolve_project_path
from src.text_sections import PAGE_CONTENT


SAMPLE_IMAGE_DIR = Path(__file__).resolve().parents[1] / "assets" / "sample_images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def get_asset_sample_image_paths() -> list[Path]:
    if not SAMPLE_IMAGE_DIR.exists():
        return []
    return sorted(
        path for path in SAMPLE_IMAGE_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_dataset_sample_images() -> pd.DataFrame:
    sample_df = get_selectable_example_images(
        preferred_split="test",
        labels=("NORMAL", "PNEUMONIA"),
        n_per_label=2,
    ).copy()
    sample_df["resolved_path"] = sample_df["image_path"].apply(resolve_project_path)
    return sample_df


def get_available_models() -> dict[str, Path]:
    return {
        name: path
        for name, path in MODEL_REGISTRY.items()
        if Path(path).exists()
    }


def load_display_image(image_source) -> Image.Image:
    if isinstance(image_source, Path):
        return Image.open(image_source).convert("RGB")
    if hasattr(image_source, "seek"):
        image_source.seek(0)
    return Image.open(image_source).convert("RGB")


def get_selected_image_source():
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        help="Use a PNG or JPEG image for this demo.",
    )

    sample_paths = get_asset_sample_image_paths()
    selected_sample = None
    sample_help_text = None

    if sample_paths:
        sample_names = {path.name: path for path in sample_paths}
        selected_sample_name = st.selectbox(
            "Or choose a sample image",
            options=["None"] + list(sample_names.keys()),
            index=0,
        )
        if selected_sample_name != "None":
            selected_sample = sample_names[selected_sample_name]
        sample_help_text = "Showing curated local examples from `assets/sample_images/`."
    else:
        try:
            dataset_samples = get_dataset_sample_images()
        except (FileNotFoundError, ValueError):
            st.caption(
                "No local sample images were found in `assets/sample_images/`, and no downloaded dataset examples are available yet. "
                "Upload an image or run `python scripts/download_data.py` to unlock dataset-backed examples."
            )
        else:
            sample_names = {
                f"{row.label.title()} | {row.source_split} | {row.filename}": row.resolved_path
                for row in dataset_samples.itertuples(index=False)
            }
            selected_sample_name = st.selectbox(
                "Or choose a sample image",
                options=["None"] + list(sample_names.keys()),
                index=0,
            )
            if selected_sample_name != "None":
                selected_sample = sample_names[selected_sample_name]
            sample_help_text = (
                "No curated local samples were found, so this page is using a few downloaded dataset examples "
                "from the test split when available."
            )

    if sample_help_text:
        st.caption(sample_help_text)

    return uploaded_file or selected_sample


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Prediction Lab",
    title="Test the model on a single chest X-ray",
    description=(
        "This page is designed to help users understand model behavior, not to provide clinical guidance."
    ),
    chips=["Single-image inference", "Threshold control", "Model comparison", "Explainability overlay"],
)

available_models = get_available_models()
if not available_models:
    missing_files = ", ".join(str(path.name) for path in MODEL_REGISTRY.values())
    render_surface_card(
        eyebrow="Setup Status",
        title="Prediction is waiting for a trained model",
        body=(
            f"Missing file(s): {missing_files}\n"
            "Next step: run `python src/train.py --epochs 1` to create `models/baseline_model.pt`.\n"
            "After training, this page will enable single-image inference and explainability."
        ),
    )
else:
    control_left, control_right, control_aux = st.columns([1.1, 1.1, 0.8], gap="large")
    with control_left:
        image_source = get_selected_image_source()
    with control_right:
        model_labels = list(available_models.keys())
        selected_model = st.radio(
            "Choose a model",
            options=model_labels,
            index=model_labels.index("baseline") if "baseline" in model_labels else 0,
            horizontal=True,
            format_func=lambda value: "Baseline model" if value == "baseline" else "Augmented model",
        )
    with control_aux:
        threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Treat the Pneumonia probability as positive only if it meets or exceeds this threshold.",
        )
        st.caption(
            "How threshold affects the output:\n"
            "A lower threshold makes the model more likely to classify an image as Pneumonia, while a higher threshold makes the model more conservative. "
            "This changes the balance between catching more true cases and generating more false positives."
        )

    if image_source is None:
        render_surface_card(
            eyebrow="Ready When You Are",
            title="Upload an image or add a sample file",
            body=(
                "The Prediction Lab supports either a user upload, a curated sample image from `assets/sample_images/`, "
                "or a small set of dataset-backed examples when the Kaggle data has already been downloaded. "
                "Once an image is selected, the page will run inference and display the result."
            ),
        )
    else:
        try:
            preview_image = load_display_image(image_source)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
            render_surface_card(
                eyebrow="Image Issue",
                title="The selected image could not be read",
                body=(
                    "Please use a readable PNG or JPEG chest X-ray image.\n"
                    f"Current message: {exc}"
                ),
            )
        else:
            selected_model_path = available_models[selected_model]

            try:
                prediction = predict_image(image_source, model_path=selected_model_path)
            except (FileNotFoundError, ValueError, RuntimeError, ModuleNotFoundError) as exc:
                render_surface_card(
                    eyebrow="Setup Status",
                    title="Prediction could not run yet",
                    body=(
                        f"Missing or unavailable file: {Path(selected_model_path).name}\n"
                        "Next step: run `python src/train.py --epochs 1` and then retry this page.\n"
                        f"Current message: {exc}"
                    ),
                )
            else:
                prob_pneumonia = prediction["class_probabilities"].get("PNEUMONIA", 0.0)
                threshold_label = "PNEUMONIA" if prob_pneumonia >= threshold else "NORMAL"

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                image_col, result_col = st.columns([0.92, 1.08], gap="large")
                with image_col:
                    st.markdown("### Selected image")
                    st.image(preview_image, use_column_width=True)
                    st.caption("This preview is the image passed into the selected model path.")

                with result_col:
                    metric1, metric2 = st.columns(2)
                    metric1.metric("Top predicted class", prediction["predicted_class"].title())
                    metric2.metric("Top class probability", f"{prediction['predicted_probability']:.1%}")

                    metric3, metric4 = st.columns(2)
                    metric3.metric("Pneumonia probability", f"{prob_pneumonia:.1%}")
                    metric4.metric("Threshold-based label", threshold_label.title())

                    st.info(
                        "Important Reminder:\n"
                        "This page is designed to help users understand model behavior on a public dataset. It is not a diagnostic system and should not be used to make medical decisions."
                    )

                    render_surface_card(
                        eyebrow="Interpretation",
                        title="How thresholding changes the label",
                        body=(
                            f"The model assigns a Pneumonia probability of {prob_pneumonia:.1%}. "
                            f"With the current threshold set to {threshold:.0%}, the displayed classification becomes {threshold_label.title()}. "
                            "This is a simple way to illustrate how operating thresholds influence model behavior."
                        ),
                    )

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                prob_left, prob_right = st.columns([1.0, 1.0], gap="large")
                with prob_left:
                    render_surface_card(
                        eyebrow="Prediction Summary",
                        title="What the model is saying",
                        body=(
                            f"Selected model: {'Baseline' if selected_model == 'baseline' else 'Augmented'}\n"
                            f"Predicted class: {prediction['predicted_class'].title()}\n"
                            f"Prediction confidence: {prediction['predicted_probability']:.1%}\n"
                            f"Pneumonia probability at the current threshold view: {prob_pneumonia:.1%}"
                        ),
                    )
                with prob_right:
                    render_surface_card(
                        eyebrow="Notes",
                        title="How to read this demo",
                        body=PAGE_CONTENT["prediction_lab"]["notes"],
                    )

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                render_surface_card(
                    eyebrow="How to read this output",
                    title="How to read this output",
                    body=(
                        "The confidence score reflects how strongly the model leans toward one class based on patterns learned from the training data. "
                        "It is not the same as certainty, and higher confidence does not make a prediction clinically valid."
                    ),
                )

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                support_left, support_mid, support_right = st.columns(3, gap="large")
                with support_left:
                    render_surface_card(
                        eyebrow="Key Terms",
                        title="Key Terms",
                        body=(
                            "Think of the threshold as a decision line. If the Pneumonia score crosses that line, the output flips to Pneumonia. "
                            "Confidence is the model's score, not a guarantee that the prediction is correct."
                        ),
                    )
                with support_mid:
                    render_surface_card(
                        eyebrow="Why this matters",
                        title="Why this matters",
                        body=(
                            "In a screening-style setting, catching more possible pneumonia cases can be helpful, but it often means flagging more normal images too. "
                            "That tradeoff is why threshold decisions matter in practice."
                        ),
                    )
                with support_right:
                    render_surface_card(
                        eyebrow="What this output does not mean",
                        title="What this output does not mean",
                        body=(
                            "This output does not confirm that a patient has or does not have pneumonia. "
                            "It is a model demonstration built on public data and should not be used as medical advice or clinical guidance."
                        ),
                    )

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                st.markdown("### Explainability view")
                explanation = try_explain_prediction(image_source, model_path=selected_model_path)
                if explanation.get("explanation_available"):
                    exp_left, exp_right = st.columns(2, gap="large")
                    with exp_left:
                        st.image(explanation["overlay_image"], use_column_width=True)
                        st.caption("Overlay image: the original chest X-ray blended with a lightweight Grad-CAM style attention map.")
                    with exp_right:
                        st.image(explanation["heatmap_image"], use_column_width=True)
                        st.caption("Heatmap image: the raw activation view before it is blended back onto the image.")
                    st.markdown(
                        "What this view shows: brighter regions highlight parts of the image that contributed more strongly to the current prediction."
                    )
                    st.markdown(
                        "How to interpret it: this is a lightweight visualization aid for model behavior, not a diagnostic explanation or a substitute for clinical review."
                    )
                else:
                    render_surface_card(
                        eyebrow="Explainability",
                        title="Explanation overlay unavailable",
                        body=(
                            "The prediction ran, but a Grad-CAM style explanation image could not be generated for this input or model configuration.\n"
                            f"Current message: {explanation.get('reason', explanation.get('explanation_note', 'Explanation unavailable.'))}"
                        ),
                    )

                st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

                render_surface_card(
                    eyebrow="Context",
                    title="Why this page matters",
                    body=(
                        "The Prediction Lab helps bridge model training and user experience. It shows how a technical pipeline can be turned into "
                        "a more understandable interface, complete with confidence, thresholding, and optional visual explanation."
                    ),
                )
