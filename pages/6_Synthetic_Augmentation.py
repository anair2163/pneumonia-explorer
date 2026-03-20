from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.charts import build_metric_comparison_chart, build_probability_distribution_histogram
from src.evaluate import MODEL_REGISTRY
from src.explain import get_augmentation_notes
from src.layout import render_page_header, render_sidebar, render_surface_card, setup_page
from src.text_sections import PAGE_CONTENT


@st.cache_data(show_spinner=False)
def load_model_summaries() -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for experiment_name, entry in MODEL_REGISTRY.items():
        summary_path = Path(entry["summary_path"])
        predictions_path = Path(entry["predictions_path"])

        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
        summaries[experiment_name] = {
            "summary": summary,
            "predictions": predictions,
        }
    return summaries


def format_metric_delta(baseline_value, augmented_value) -> str:
    if baseline_value is None or augmented_value is None or pd.isna(baseline_value) or pd.isna(augmented_value):
        return "N/A"
    delta = float(augmented_value) - float(baseline_value)
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def format_metric_value(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def render_comparison_metric_cards(baseline_summary: dict, augmented_summary: dict) -> None:
    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    metric_cols = st.columns(5, gap="large")
    for idx, (label, key) in enumerate(metric_labels):
        baseline_value = baseline_summary.get(key)
        augmented_value = augmented_summary.get(key)
        metric_cols[idx].metric(
            label,
            format_metric_value(augmented_value),
            delta=format_metric_delta(baseline_value, augmented_value),
        )
        metric_cols[idx].caption(f"Baseline: {format_metric_value(baseline_value)}")


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Synthetic Augmentation",
    title="Comparing the baseline model with an augmentation experiment",
    description=(
        "This section compares a baseline training setup with a stronger augmentation-based training configuration. "
        "The purpose is to test whether augmentation improves generalization in this dataset, not to assume that more augmentation automatically leads to better medical image models."
    ),
    chips=["Baseline vs augmented", "Metric comparison", "Experimental setup", "Educational use only"],
)

summaries = load_model_summaries()
baseline = summaries.get("baseline")
augmented = summaries.get("augmented")

if baseline is None:
    render_surface_card(
        eyebrow="Setup Status",
        title="The augmentation comparison needs a baseline run first",
        body=(
            f"Missing file: {Path(MODEL_REGISTRY['baseline']['summary_path']).name}\n"
            "Next step: run `python src/train.py --epochs 1` and then `python src/evaluate.py`.\n"
            "Once the baseline artifacts exist, this page can anchor the augmentation comparison."
        ),
    )
elif augmented is None:
    placeholder_left, placeholder_right = st.columns([1.1, 0.9], gap="large")
    with placeholder_left:
        render_surface_card(
            eyebrow="Setup Status",
            title="The augmented experiment has not been evaluated yet",
            body=(
                f"Expected files: {Path(MODEL_REGISTRY['augmented']['model_path']).name}, "
                f"{Path(MODEL_REGISTRY['augmented']['summary_path']).name}\n"
                "Next step: run `python src/train.py --experiment augmented --epochs 1` and then `python src/evaluate.py`.\n"
                "Once the augmented artifacts exist, this page will switch to a full side-by-side comparison."
            ),
        )
    with placeholder_right:
        render_surface_card(
            eyebrow="Important Note",
            title="Important Note",
            body=(
                "Augmentation in this project is an experimental modeling strategy, not a substitute for broader clinical data diversity. "
                "Even if augmentation changes performance, that does not mean the model is ready for real-world medical use."
            ),
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    render_surface_card(
        eyebrow="Why This Still Matters",
        title="Why stronger augmentation may help and why it may not",
        body=(
            "Stronger augmentation can help a model see more varied versions of the same class patterns, which may improve robustness. "
            "But any gain can be dataset-specific, and offline improvements do not automatically mean better generalization in different clinical settings."
        ),
    )
else:
    baseline_summary = baseline["summary"]
    augmented_summary = augmented["summary"]
    baseline_better_metrics = [
        metric_name
        for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc")
        if baseline_summary.get(metric_name) is not None
        and augmented_summary.get(metric_name) is not None
        and float(baseline_summary[metric_name]) > float(augmented_summary[metric_name])
    ]

    comparison_df = pd.DataFrame(
        [
            {
                "model": "Baseline",
                "accuracy": baseline_summary.get("accuracy"),
                "precision": baseline_summary.get("precision"),
                "recall": baseline_summary.get("recall"),
                "f1": baseline_summary.get("f1"),
                "roc_auc": baseline_summary.get("roc_auc"),
            },
            {
                "model": "Augmented",
                "accuracy": augmented_summary.get("accuracy"),
                "precision": augmented_summary.get("precision"),
                "recall": augmented_summary.get("recall"),
                "f1": augmented_summary.get("f1"),
                "roc_auc": augmented_summary.get("roc_auc"),
            },
        ]
    )

    render_comparison_metric_cards(baseline_summary, augmented_summary)

    st.caption(
        "Each card shows the augmented-model result, with the delta versus baseline and the baseline value listed underneath. "
        "This makes the comparison easy to scan without losing the original reference point."
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    render_surface_card(
        eyebrow="Current Readout",
        title="In this version, augmentation did not improve the baseline",
        body=(
            "In the current experiment, the augmented model did not outperform the baseline model on the main evaluation metrics. "
            "That does not mean augmentation is never useful, but it does show that augmentation strategies need to be validated empirically rather than assumed to help by default.\n"
            f"Baseline currently leads on: {', '.join(metric.upper() for metric in baseline_better_metrics) if baseline_better_metrics else 'the available saved metrics'}."
        ),
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    top_left, top_right = st.columns([1.05, 0.95], gap="large")
    with top_left:
        st.markdown("### Metric comparison")
        st.plotly_chart(
            build_metric_comparison_chart(comparison_df, title="Baseline vs augmented performance"),
            use_container_width=True,
        )
        render_surface_card(
            eyebrow="Interpretation",
            title="How to read the augmentation comparison",
            body=(
                "What this shows:\n"
                "This chart compares the baseline and augmented models across the main evaluation metrics.\n\n"
                "How to interpret it:\n"
                "It highlights whether augmentation changed performance meaningfully across accuracy, precision, recall, F1, and ROC-AUC rather than focusing on a single number.\n\n"
                "Why it matters:\n"
                "Augmentation is often expected to improve robustness, but it does not automatically improve results. Comparing models directly helps show whether the added complexity produced a measurable benefit in this project."
            ),
        )
    with top_right:
        render_surface_card(
            eyebrow="Important Note",
            title="Important Note",
            body=(
                "Augmentation in this project is an experimental modeling strategy, not a substitute for broader clinical data diversity. "
                "Even if augmentation changes performance, that does not mean the model is ready for real-world medical use."
            ),
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.0, 1.0], gap="large")
    with bottom_left:
        st.markdown("### Confidence distribution comparison")
        if baseline["predictions"] is not None and augmented["predictions"] is not None:
            baseline_hist_df = baseline["predictions"].copy()
            baseline_hist_df["model"] = "Baseline"
            augmented_hist_df = augmented["predictions"].copy()
            augmented_hist_df["model"] = "Augmented"
            combined_hist_df = pd.concat([baseline_hist_df, augmented_hist_df], ignore_index=True)

            st.plotly_chart(
                build_probability_distribution_histogram(
                    combined_hist_df,
                    color_col="model",
                    title="Pneumonia probability distributions by model",
                ),
                use_container_width=True,
            )
            st.markdown(
                "What this chart shows: It compares how the baseline and augmented models distribute Pneumonia probabilities across the saved test predictions."
            )
            st.markdown(
                "How to interpret it: Cleaner separation can indicate that one model is assigning more decisive scores, while overlapping distributions suggest more ambiguous confidence."
            )
            st.markdown(
                "Why it matters here: A model that improves metrics while also producing more coherent confidence patterns is generally easier to reason about in a demo or decision-support setting."
            )
        else:
            render_surface_card(
                eyebrow="Probability Comparison",
                title="Prediction distribution data not available",
                body=(
                    "Per-image prediction outputs were not found for both models, so the confidence distribution comparison is not available yet.\n"
                    "Next step: run `python src/evaluate.py` after both checkpoints are available."
                ),
            )

    with bottom_right:
        render_surface_card(
            eyebrow="Interpretation",
            title="Why the augmented run may not have helped",
            body=(
                "Several practical explanations fit the current result.\n"
                "The baseline is already strong, so there may be limited room for simple augmentation to add value.\n"
                "The chosen transforms may not match the visual structure of this task especially well.\n"
                "With a short training schedule, the augmented model may need more tuning time before any robustness benefit appears."
            ),
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    closing_left, closing_right = st.columns([1.0, 1.0], gap="large")
    with closing_left:
        render_surface_card(
            eyebrow="Why It May Help",
            title="What a positive result would suggest",
            body=(
                "If the augmented model improves recall, F1, or ROC-AUC without harming precision too much, that suggests the stronger transformations may be helping the model learn a more resilient decision boundary."
            ),
        )
    with closing_right:
        render_surface_card(
            eyebrow="Next Step",
            title="What to validate before claiming success",
            body=PAGE_CONTENT["synthetic_augmentation"]["next_steps"],
        )
