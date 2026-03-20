import json
from pathlib import Path

import streamlit as st

from src.layout import (
    render_home_hero,
    render_info_strip,
    render_page_header,
    render_profile_links_card,
    render_sidebar,
    render_surface_card,
    setup_page,
)


BASELINE_SUMMARY_PATH = Path("models/baseline_eval_summary.json")
AUGMENTED_SUMMARY_PATH = Path("models/aug_eval_summary.json")


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def format_summary_value(value, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    numeric = float(value)
    return f"{numeric:.1%}" if percent else f"{numeric:.3f}"


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Project Focus",
    title="Pneumonia Detection & Synthetic Augmentation Explorer",
    description=(
        "Built by Ahan Nair, this project explores how computer vision models detect pneumonia from chest X-rays, "
        "how prediction behavior changes across thresholds and evaluation metrics, and how augmentation can affect performance."
    ),
    chips=[
        "Interactive analytics",
        "Prediction thresholds",
        "Evaluation tradeoffs",
        "Augmentation comparison",
    ],
)

hero_col, hero_side = st.columns([1.45, 0.85], gap="large")
baseline_summary = load_summary(BASELINE_SUMMARY_PATH)
augmented_summary = load_summary(AUGMENTED_SUMMARY_PATH)

summary_items = [
    ("Project at a glance", "Binary chest X-ray classification with side-by-side model review."),
    ("What users can explore", "Threshold behavior, confidence, metrics, and augmentation tradeoffs."),
    ("Use boundary", "Educational demo only, not a medical or diagnostic tool."),
]

if baseline_summary is not None:
    summary_items = [
        ("Baseline recall", format_summary_value(baseline_summary.get("recall"), percent=True)),
        ("Baseline ROC-AUC", format_summary_value(baseline_summary.get("roc_auc"))),
        (
            "Augmentation outcome",
            (
                "Baseline still leads"
                if augmented_summary is not None
                and baseline_summary.get("f1") is not None
                and augmented_summary.get("f1") is not None
                and float(baseline_summary.get("f1")) >= float(augmented_summary.get("f1"))
                else "Awaiting comparison"
            ),
        ),
    ]

with hero_col:
    render_home_hero(
        title="Why this project was built",
        body=(
            "Chest X-rays are one of the most widely used imaging tools for evaluating lung conditions, but model performance depends on more than just accuracy. "
            "Threshold choice, class balance, augmentation strategy, and error tradeoffs all shape how useful a system may be in practice. "
            "This project was built to make those dynamics easier to explore through interactive analytics, model outputs, and side-by-side evaluation views."
        ),
    )
    render_info_strip(
        items=summary_items
    )
    st.caption(
        "This project combines dataset exploration, interactive prediction, and model evaluation to show how pneumonia detection models behave beyond a single accuracy number. "
        "The goal is to make threshold tradeoffs, error patterns, and augmentation effects easier to understand in one place."
    )

with hero_side:
    render_surface_card(
        eyebrow="Mini Visual Summary",
        title="What these results suggest",
        body=(
            "The baseline model performs like a screening-oriented classifier, with very high recall and strong class separation, but also a meaningful number of false positives. "
            "The comparison view helps show that augmentation did not improve performance in the current setup."
        ),
    )
    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)
    render_profile_links_card(eyebrow="Links", title="Project links")

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

render_surface_card(
    eyebrow="Use Guidance",
    title="How this site should be used",
    body=(
        "This website is an educational machine learning project built using public chest X-ray data. "
        "It is not a medical device, not a diagnostic system, and not a substitute for professional medical judgment. "
        "The predictions, visualizations, and explanations shown here are for learning and exploration only and should not be used to make clinical decisions."
    ),
)

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

warn1, warn2, warn3 = st.columns(3, gap="large")

with warn1:
    render_surface_card(
        eyebrow="Boundary",
        title="Educational scope only",
        body="The outputs here are technical demonstrations meant to help users understand model behavior on a public dataset.",
    )

with warn2:
    render_surface_card(
        eyebrow="Reminder",
        title="Confidence is not certainty",
        body="A high confidence score reflects the model's internal signal strength, not a clinically reliable conclusion.",
    )

with warn3:
    render_surface_card(
        eyebrow="Context",
        title="Public data has limits",
        body="Public benchmark results can be informative for learning, but they do not establish readiness for real-world medical deployment.",
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    render_surface_card(
        eyebrow="Builder",
        title="About Ahan Nair",
        body=(
            "I’m Ahan Nair, a data science and analytics-focused builder interested in designing interactive tools that make complex model behavior easier to understand. "
            "I’m especially interested in projects that combine machine learning, visual communication, and practical decision support in ways that are both technically grounded and easy to explore."
        ),
    )

with col2:
    render_surface_card(
        eyebrow="Case Study",
        title="Why pneumonia detection is a useful lens",
        body=(
            "Pneumonia detection is a useful case study because it sits at the intersection of computer vision, healthcare, and model evaluation. "
            "Rather than focusing only on whether a classifier works, this project looks at how it behaves: how performance changes across thresholds, where errors happen, and whether augmentation helps improve results."
        ),
    )

with col3:
    render_surface_card(
        eyebrow="Use Boundary",
        title="What responsible use looks like",
        body=(
            "This project is intended for educational and technical exploration only. It demonstrates how a deep learning workflow can be trained, evaluated, and interpreted on a public dataset, "
            "but it is not intended for diagnosis, treatment, screening decisions, or any form of clinical use. Any real medical interpretation should be performed by qualified healthcare professionals."
        ),
    )

st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    render_surface_card(
        eyebrow="Site Map",
        title="What this site includes",
        body=(
            "- Interactive analytics to explore class balance, dataset structure, and model behavior\n"
            "- Prediction tools that show how confidence and threshold settings affect outputs\n"
            "- Performance review pages that explain model metrics, error tradeoffs, and augmentation results"
        ),
    )

with right:
    render_surface_card(
        eyebrow="Use Reminder",
        title="What to keep in mind while exploring",
        body=(
            "Use the pages together rather than in isolation: the data view sets context, the prediction view shows threshold effects, "
            "and the performance pages explain how those outputs should be interpreted."
        ),
    )

st.markdown(
    "Built by Ahan Nair  \nEducational machine learning project - not for clinical or diagnostic use"
)
