from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluate import MODEL_REGISTRY
from src.layout import (
    render_page_header,
    render_profile_links_card,
    render_sidebar,
    render_surface_card,
    setup_page,
)
from src.text_sections import get_text_block


def load_summary(experiment_name: str) -> dict | None:
    summary_path = Path(MODEL_REGISTRY[experiment_name]["summary_path"])
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def load_baseline_artifacts() -> tuple[dict | None, pd.DataFrame | None]:
    baseline_entry = MODEL_REGISTRY["baseline"]
    summary_path = Path(baseline_entry["summary_path"])
    predictions_path = Path(baseline_entry["predictions_path"])

    if not summary_path.exists():
        return None, None

    summary = json.loads(summary_path.read_text())
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
    return summary, predictions


def format_metric(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def build_dynamic_findings(
    baseline_summary: dict,
    baseline_predictions: pd.DataFrame | None,
    augmented_summary: dict | None,
) -> tuple[list[str], list[str], list[str]]:
    recall_value = format_metric(baseline_summary.get("recall"))
    roc_auc_value = format_metric(baseline_summary.get("roc_auc"))
    accuracy_value = format_metric(baseline_summary.get("accuracy"))
    f1_value = format_metric(baseline_summary.get("f1"))

    confusion = baseline_summary.get("confusion_matrix", {})
    confusion_values = confusion.get("values", [[None, None], [None, None]])
    false_positives = confusion_values[0][1]
    false_negatives = confusion_values[1][0]
    aug_accuracy = format_metric(augmented_summary.get("accuracy")) if augmented_summary is not None else "N/A"
    aug_f1 = format_metric(augmented_summary.get("f1")) if augmented_summary is not None else "N/A"
    aug_roc_auc = format_metric(augmented_summary.get("roc_auc")) if augmented_summary is not None else "N/A"

    findings = [
        f"The baseline model achieved very high recall ({recall_value}) and strong ROC-AUC ({roc_auc_value}), while also reaching accuracy of {accuracy_value} and F1 of {f1_value} on the untouched test split.",
        f"The current models behave like screening-oriented classifiers: they miss very few pneumonia cases ({false_negatives} false negative for the baseline) but accept many false positives ({false_positives} Normal scans flagged as Pneumonia).",
        f"The baseline slightly outperformed the augmented model in this experiment, with stronger accuracy ({accuracy_value} vs {aug_accuracy}), F1 ({f1_value} vs {aug_f1}), and ROC-AUC ({roc_auc_value} vs {aug_roc_auc}).",
        "Synthetic augmentation did not improve results in the current configuration, so it should be treated as a comparison result rather than as a gain.",
    ]

    takeaways = [
        "This kind of model may be more useful for triage or screening support than for final diagnosis, because it is tuned toward sensitivity rather than selectivity.",
        "Threshold choice matters because the operating point changes the balance between catching more pneumonia cases and reducing false positives.",
        "Augmentation strategies should be validated empirically rather than assumed to help, especially when the baseline is already strong.",
    ]

    limitations = [
        "The results are limited by dataset scope, class balance, and the representativeness of the available train and test splits.",
        "Medical imaging predictions shown in this app are not medical advice and should not be treated as diagnostic output.",
        "One training setup is not enough to fully judge augmentation effectiveness, so broader tuning and validation would still be needed.",
    ]

    return findings, takeaways, limitations


PLACEHOLDER_FINDINGS = [
    "The project already demonstrates strong structure: data setup, modeling modules, evaluation scaffolding, and a polished multipage app experience.",
    "The app is designed to present machine learning work as a reviewable product case study rather than a notebook-only artifact.",
    "The augmentation path creates a clear comparison framework for future baseline-versus-augmented evaluation once model artifacts are generated.",
    "The current setup messaging makes the project easier to run locally and easier to explain during a walkthrough of the workflow.",
]

PLACEHOLDER_TAKEAWAYS = [
    "A clean interface and readable evaluation story can make technical work much easier for non-specialists to understand.",
    "Project readiness matters: dataset download, training, evaluation, and app launch should feel like one coherent workflow.",
    "Even before final metrics exist, the structure already shows practical engineering judgment and communication skill.",
]

PLACEHOLDER_LIMITATIONS = [
    "No evaluated model artifacts are available yet, so this page cannot summarize measured performance.",
    "Any future results will still need to be interpreted in the context of dataset scope, label quality, and generalization risk.",
    "This work is an educational machine learning project, not a clinical decision tool.",
]

FUTURE_IMPROVEMENTS = [
    "Train and evaluate both baseline and augmented models, then promote the strongest saved result into this summary automatically.",
    "Add richer error analysis, calibration review, and a small set of representative success and failure cases.",
    "Validate the workflow on additional external data before making stronger generalization claims.",
]


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Key Findings",
    title="Executive summary of the project and what it demonstrates",
    description=(
        "This project combines exploratory analytics, image classification, and model comparison to better understand how pneumonia detection systems behave on a public chest X-ray dataset. "
        "The focus is on interpretation, tradeoffs, and responsible presentation rather than clinical deployment claims."
    ),
    chips=["Executive summary", "Model tradeoffs", "Educational use only", "Next steps"],
)

baseline_summary, baseline_predictions = load_baseline_artifacts()
augmented_summary = load_summary("augmented")

if baseline_summary is not None:
    findings, takeaways, limitations = build_dynamic_findings(
        baseline_summary,
        baseline_predictions,
        augmented_summary,
    )

    render_surface_card(
        eyebrow="About this Project",
        title="About this Project",
        body=(
            "I built this project to explore how healthcare machine learning systems can be made more interactive and interpretable. "
            "Instead of presenting only final metrics, the site is designed to help users examine the data, test predictions, compare model variants, and understand the tradeoffs behind performance."
        ),
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    metric_cols = st.columns(4, gap="large")
    metric_cols[0].metric("Baseline Accuracy", format_metric(baseline_summary.get("accuracy")))
    metric_cols[1].metric("Baseline Recall", format_metric(baseline_summary.get("recall")))
    metric_cols[2].metric("Baseline F1", format_metric(baseline_summary.get("f1")))
    metric_cols[3].metric("Baseline ROC-AUC", format_metric(baseline_summary.get("roc_auc")))

    st.caption(
        "These cards reflect the saved baseline evaluation summary currently available in the project artifacts."
    )
else:
    findings = PLACEHOLDER_FINDINGS
    takeaways = PLACEHOLDER_TAKEAWAYS
    limitations = PLACEHOLDER_LIMITATIONS

    render_surface_card(
        eyebrow="Project Summary",
        title="Findings will become more specific after evaluation artifacts are generated",
        body=(
            "This project already has the structure for a strong end-to-end case study, but the final measured findings depend on saved model evaluation outputs.\n"
            "Next steps: run `python src/train.py --epochs 1` and then `python src/evaluate.py`.\n"
            "Once those artifacts exist, this page will summarize the best available saved model result automatically."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

top_left, top_right = st.columns([1.0, 1.0], gap="large")
with top_left:
    st.markdown("### Key findings")
    for finding in findings:
        st.write(f"- {finding}")
with top_right:
    st.markdown("### Practical takeaways")
    for takeaway in takeaways:
        st.write(f"- {takeaway}")

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

bottom_left, bottom_right = st.columns([1.0, 1.0], gap="large")
with bottom_left:
    st.markdown("### Limitations")
    for item in limitations:
        st.write(f"- {item}")
with bottom_right:
    st.markdown("### Future improvements")
    for item in FUTURE_IMPROVEMENTS:
        st.write(f"- {item}")

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

link_left, link_right = st.columns([1.0, 1.0], gap="large")
with link_left:
    render_profile_links_card(eyebrow="Profile Links", title="Connect")
with link_right:
    render_surface_card(
        eyebrow="Educational Use Only",
        title="Educational Use Only",
        body=(
            "This website is an educational project and is not intended for diagnosis, treatment, screening, or any other medical use. "
            "All outputs should be interpreted as technical demonstrations only."
        ),
    )
