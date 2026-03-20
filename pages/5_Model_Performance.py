from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.charts import (
    build_confusion_matrix_heatmap,
    build_metric_comparison_chart,
    build_probability_distribution_histogram,
    build_roc_curve,
)
from src.evaluate import MODEL_REGISTRY
from src.layout import render_page_header, render_sidebar, render_surface_card, setup_page
from src.text_sections import PAGE_CONTENT


@st.cache_data(show_spinner=False)
def load_evaluation_artifacts() -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for experiment_name, entry in MODEL_REGISTRY.items():
        summary_path = Path(entry["summary_path"])
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        predictions_df = _read_csv_if_exists(entry["predictions_path"])
        sample_predictions_df = _read_csv_if_exists(entry["sample_predictions_path"])
        roc_curve_df = _read_csv_if_exists(entry["roc_curve_path"])

        artifacts[experiment_name] = {
            "summary": summary,
            "predictions": predictions_df,
            "sample_predictions": sample_predictions_df,
            "roc_curve": roc_curve_df,
            "files": {
                "summary": summary_path,
                "predictions": Path(entry["predictions_path"]),
                "sample_predictions": Path(entry["sample_predictions_path"]),
                "roc_curve": Path(entry["roc_curve_path"]),
            },
        }
    return artifacts


def _read_csv_if_exists(path_like) -> pd.DataFrame | None:
    path = Path(path_like)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_metric(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def build_comparison_dataframe(artifacts: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for experiment_name in ("baseline", "augmented"):
        artifact = artifacts.get(experiment_name)
        if artifact is None:
            continue

        summary = artifact["summary"]
        rows.append(
            {
                "model": summary.get("experiment_name", experiment_name).title(),
                "accuracy": summary.get("accuracy"),
                "precision": summary.get("precision"),
                "recall": summary.get("recall"),
                "f1": summary.get("f1"),
                "roc_auc": summary.get("roc_auc"),
                "status": summary.get("status", "evaluated"),
            }
        )

    return pd.DataFrame(rows)


def build_missing_artifacts_message() -> tuple[str, str]:
    baseline_model_path = Path(MODEL_REGISTRY["baseline"]["model_path"])
    baseline_summary_path = Path(MODEL_REGISTRY["baseline"]["summary_path"])
    aug_model_path = Path(MODEL_REGISTRY["augmented"]["model_path"])
    aug_summary_path = Path(MODEL_REGISTRY["augmented"]["summary_path"])

    if not baseline_model_path.exists():
        return (
            "Performance review is waiting for the first trained model",
            (
                f"Missing file: {baseline_model_path.name}\n"
                "Next step 1: run `python src/train.py --epochs 1` to train the baseline model.\n"
                "Next step 2: run `python src/evaluate.py` to generate review artifacts for this page."
            ),
        )

    if not baseline_summary_path.exists():
        return (
            "Performance review is waiting for evaluation artifacts",
            (
                f"Missing file: {baseline_summary_path.name}\n"
                "Next step: run `python src/evaluate.py` to generate the saved metrics, confusion matrix summary, "
                "and per-image prediction outputs for the baseline model."
            ),
        )

    if aug_model_path.exists() and not aug_summary_path.exists():
        return (
            "The augmented model has not been evaluated yet",
            (
                f"Missing file: {aug_summary_path.name}\n"
                "Next step: run `python src/evaluate.py` to add the augmented model artifacts and enable side-by-side review."
            ),
        )

    expected_files = ", ".join(
        str(Path(entry["summary_path"]).name) for entry in MODEL_REGISTRY.values()
    )
    return (
        "Performance review is waiting for evaluation artifacts",
        (
            f"Expected summary files: {expected_files}\n"
            "Next step: run `python src/evaluate.py` after training at least the baseline model."
        ),
    )


def get_metric_winner_rows(comparison_df: pd.DataFrame) -> list[dict[str, str]]:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows: list[dict[str, str]] = []

    for metric_name in metric_names:
        metric_df = comparison_df.dropna(subset=[metric_name])
        if metric_df.empty:
            rows.append({"label": metric_name.upper(), "winner": "N/A", "value": "N/A", "delta": "N/A"})
            continue

        best_idx = metric_df[metric_name].idxmax()
        best_row = metric_df.loc[best_idx]
        sorted_values = metric_df[metric_name].sort_values(ascending=False).tolist()
        delta = sorted_values[0] - sorted_values[1] if len(sorted_values) > 1 else 0.0
        rows.append(
            {
                "label": metric_name.upper(),
                "winner": str(best_row["model"]),
                "value": _format_metric(best_row[metric_name]),
                "delta": f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
            }
        )

    return rows


def build_combined_roc_chart(artifacts: dict[str, dict]):
    fig = go.Figure()
    color_map = {"baseline": "#1d4ed8", "augmented": "#0f766e"}

    for experiment_name in ("baseline", "augmented"):
        artifact = artifacts.get(experiment_name)
        if artifact is None:
            continue

        roc_df = artifact.get("roc_curve")
        roc_auc = artifact["summary"].get("roc_auc")
        if roc_df is None or roc_df.empty:
            continue

        label = "Baseline" if experiment_name == "baseline" else "Augmented"
        auc_suffix = f" (AUC = {float(roc_auc):.3f})" if roc_auc is not None else ""
        fig.add_trace(
            go.Scatter(
                x=roc_df["false_positive_rate"],
                y=roc_df["true_positive_rate"],
                mode="lines",
                line=dict(color=color_map[experiment_name], width=3),
                name=f"{label}{auc_suffix}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#64748b", dash="dash"),
            name="Chance",
        )
    )
    fig.update_layout(
        title="ROC Curve Comparison",
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Model Performance",
    title="Reviewing model behavior through evaluation metrics and tradeoffs",
    description=(
        "This page focuses on how the model performs as a classification system, not just whether it makes correct predictions overall. "
        "Metrics such as recall, precision, and ROC-AUC help show how the model handles tradeoffs between catching pneumonia cases and incorrectly flagging normal images."
    ),
    chips=["Evaluation artifacts", "Metric tradeoffs", "ROC analysis", "Educational use only"],
)

artifacts = load_evaluation_artifacts()

if not artifacts:
    title, body = build_missing_artifacts_message()
    render_surface_card(
        eyebrow="Setup Status",
        title=title,
        body=body,
    )
else:
    comparison_df = build_comparison_dataframe(artifacts)
    available_model_keys = [key for key in ("baseline", "augmented") if key in artifacts]
    metric_winners = get_metric_winner_rows(comparison_df) if not comparison_df.empty else []

    review_left, review_right = st.columns([1.15, 0.85], gap="large")
    with review_left:
        st.markdown("### Cross-model comparison")
        if not comparison_df.empty:
            st.plotly_chart(
                build_metric_comparison_chart(comparison_df, title="Baseline vs Augmented Metrics"),
                use_container_width=True,
            )
            st.markdown(
                "What this chart shows: It compares the saved evaluation metrics for each available model using the exact JSON artifacts generated by the evaluation script."
            )
            st.markdown(
                "How to interpret it: Look for broad improvement across metrics rather than one isolated high score. A model that gains recall while sharply losing precision may still be harder to trust."
            )
            st.markdown(
                "Why it matters in pneumonia detection: Offline review should reflect both detection strength and error tradeoffs, since false negatives and false positives have different practical consequences."
            )
        else:
            render_surface_card(
                eyebrow="Comparison",
                title="Cross-model comparison is not available yet",
                body=(
                    "At least one evaluated model summary is required before a side-by-side comparison can be shown."
                ),
            )
    with review_right:
        render_surface_card(
            eyebrow="Important Context",
            title="Educational use only",
            body=(
                "Evaluation results shown here describe model behavior on a public dataset. "
                "They are useful for understanding technical performance, but they do not establish clinical safety or medical reliability."
            ),
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    if metric_winners:
        winner_cols = st.columns(5, gap="large")
        for column, winner_row in zip(winner_cols, metric_winners):
            column.metric(f"Best {winner_row['label']}", winner_row["winner"], delta=winner_row["delta"])
            column.caption(f"Top score: {winner_row['value']}")

        st.caption(
            "These cards show which saved model currently leads on each metric and by how much, making the comparison easier to scan at a glance."
        )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    if available_model_keys:
        selected_model = st.radio(
            "Select a model for detailed review",
            options=available_model_keys,
            index=available_model_keys.index("baseline") if "baseline" in available_model_keys else 0,
            horizontal=True,
            format_func=lambda value: "Baseline model" if value == "baseline" else "Augmented model",
        )

        selected_artifact = artifacts[selected_model]
        selected_summary = selected_artifact["summary"]
        selected_predictions = selected_artifact["predictions"]
        selected_roc_curve = selected_artifact["roc_curve"]

        metric_cols = st.columns(5, gap="large")
        metric_cols[0].metric("Accuracy", _format_metric(selected_summary.get("accuracy")))
        metric_cols[1].metric("Precision", _format_metric(selected_summary.get("precision")))
        metric_cols[2].metric("Recall", _format_metric(selected_summary.get("recall")))
        metric_cols[3].metric("F1", _format_metric(selected_summary.get("f1")))
        metric_cols[4].metric("ROC-AUC", _format_metric(selected_summary.get("roc_auc")))

        render_surface_card(
            eyebrow="Metric Guidance",
            title="How to read these metrics",
            body=(
                "Accuracy shows the overall share of correct predictions, but it does not explain what kinds of mistakes the model makes. "
                "Recall is especially important in screening-style tasks because it reflects how many true pneumonia cases are identified. "
                "Precision helps show how often pneumonia predictions are actually correct. "
                "ROC-AUC summarizes how well the model separates the two classes across many thresholds."
            ),
        )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        top_left, top_right = st.columns([0.95, 1.05], gap="large")
        with top_left:
            confusion = selected_summary.get("confusion_matrix")
            if confusion:
                confusion_values = confusion["values"]
                false_positives = confusion_values[0][1]
                false_negatives = confusion_values[1][0]
                st.markdown("### Confusion matrix")
                st.plotly_chart(
                    build_confusion_matrix_heatmap(
                        confusion_values,
                        labels=confusion.get("labels", ["NORMAL", "PNEUMONIA"]),
                        title=f"{selected_model.title()} confusion matrix",
                    ),
                    use_container_width=True,
                )
                render_surface_card(
                    eyebrow="Interpretation",
                    title="How to read the confusion matrix",
                    body=(
                        "What this shows:\n"
                        "The confusion matrix breaks predictions into correct and incorrect outcomes across the two classes.\n\n"
                        "How to interpret it:\n"
                        "It helps show whether the model is mostly missing pneumonia cases or mostly over-flagging normal images.\n\n"
                        "Why it matters:\n"
                        f"This is one of the clearest ways to understand the model’s error profile. For screening-style systems, it is especially useful because high sensitivity can still come with a meaningful false-positive burden. "
                        f"In this saved run, false negatives are {false_negatives} and false positives are {false_positives}."
                    ),
                )
        with top_right:
            st.markdown("### ROC curve comparison")
            if len(available_model_keys) > 1:
                st.plotly_chart(build_combined_roc_chart(artifacts), use_container_width=True)
                render_surface_card(
                    eyebrow="Interpretation",
                    title="How to read the ROC curve",
                    body=(
                        "What this shows:\n"
                        "The ROC curve shows how well the model separates Normal and Pneumonia images across many threshold settings.\n\n"
                        "How to interpret it:\n"
                        "A curve that stays closer to the top-left generally reflects stronger discrimination, while a curve closer to the diagonal suggests weaker separation.\n\n"
                        "Why it matters:\n"
                        "ROC-AUC is useful because it reflects ranking performance across thresholds rather than performance at only one cutoff. In this project, it helps show whether the model can meaningfully distinguish between the two classes on the dataset."
                    ),
                )
            elif selected_roc_curve is not None and not selected_roc_curve.empty and selected_summary.get("roc_auc") is not None:
                st.plotly_chart(
                    build_roc_curve(
                        selected_roc_curve,
                        auc_value=float(selected_summary["roc_auc"]),
                        title=f"{selected_model.title()} ROC curve",
                    ),
                    use_container_width=True,
                )
                render_surface_card(
                    eyebrow="Interpretation",
                    title="How to read the ROC curve",
                    body=(
                        "What this shows:\n"
                        "The ROC curve shows how well the model separates Normal and Pneumonia images across many threshold settings.\n\n"
                        "How to interpret it:\n"
                        "A curve that stays closer to the top-left generally reflects stronger discrimination, while a curve closer to the diagonal suggests weaker separation.\n\n"
                        "Why it matters:\n"
                        "ROC-AUC is useful because it reflects ranking performance across thresholds rather than performance at only one cutoff. In this project, it helps show whether the model can meaningfully distinguish between the two classes on the dataset."
                    ),
                )
            else:
                render_surface_card(
                    eyebrow="ROC View",
                    title="ROC data not available",
                    body=(
                        "ROC-AUC is only computed when the evaluation set contains both classes and the curve points were saved. "
                        "If that data is missing, the other metrics still provide a useful performance read."
                    ),
                )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        bottom_left, bottom_right = st.columns([1.08, 0.92], gap="large")
        with bottom_left:
            st.markdown("### Probability distribution")
            if selected_predictions is not None and not selected_predictions.empty:
                st.plotly_chart(
                    build_probability_distribution_histogram(
                        selected_predictions,
                        title=f"{selected_model.title()} probability distribution",
                    ),
                    use_container_width=True,
                )
                render_surface_card(
                    eyebrow="Interpretation",
                    title="How to read the confidence distribution",
                    body=(
                        "What this shows:\n"
                        "This view summarizes how prediction probabilities are distributed across evaluated images.\n\n"
                        "How to interpret it:\n"
                        "A wider separation between classes may suggest stronger class discrimination, while overlapping confidence distributions can indicate more ambiguous cases.\n\n"
                        "Why it matters:\n"
                        "Confidence patterns help show where threshold choice may matter most and where the model appears more or less certain."
                    ),
                )
            else:
                render_surface_card(
                    eyebrow="Probability View",
                    title="Prediction probabilities not available",
                    body=(
                        f"Missing file: {selected_artifact['files']['predictions'].name}\n"
                        "Next step: rerun `python src/evaluate.py` to regenerate the per-image prediction probabilities for this model."
                    ),
                )
        with bottom_right:
            render_surface_card(
                eyebrow="Reviewer Take",
                title="What this performance page is trying to answer",
                body=(
                    "Did the augmentation experiment improve the baseline in a meaningful way?\n"
                    "Are the gains consistent across multiple metrics?\n"
                    "Do the confusion patterns and confidence distributions suggest a model that behaves coherently rather than just achieving one strong score?"
                ),
            )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        sample_predictions = selected_artifact.get("sample_predictions")
        if sample_predictions is not None and not sample_predictions.empty:
            st.markdown("### Sample predictions")
            st.dataframe(sample_predictions, use_container_width=True, hide_index=True)
            st.caption(
                "This small table gives a concrete feel for how the model scored individual images, which helps make the evaluation outputs more tangible."
            )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        closing_left, closing_right = st.columns([1.0, 1.0], gap="large")
        with closing_left:
            render_surface_card(
                eyebrow="Performance Context",
                title="Why this review is more than a scoreboard",
                body=PAGE_CONTENT["model_performance"]["framing"],
            )
        with closing_right:
            render_surface_card(
                eyebrow="Interpretation",
                title="How to communicate results responsibly",
                body=(
                    PAGE_CONTENT["model_performance"].get("interpretation")
                    or "Results should be communicated with attention to error patterns, confidence behavior, and the limits of offline testing."
                ),
            )
    else:
        title, body = build_missing_artifacts_message()
        render_surface_card(
            eyebrow="Setup Status",
            title=title,
            body=body,
        )
