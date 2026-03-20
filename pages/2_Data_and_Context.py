from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st
from PIL import Image

from src.layout import render_page_header, render_sidebar, render_surface_card, setup_page
from src.load_data import (
    build_metadata_dataframe,
    get_selectable_example_images,
    get_dataset_overview,
    inspect_image_size_distribution,
    resolve_project_path,
)


@st.cache_data(show_spinner=False)
def load_metadata():
    metadata = build_metadata_dataframe()
    overview = get_dataset_overview()
    return metadata, overview


@st.cache_data(show_spinner=False)
def load_size_profile():
    metadata, _ = load_metadata()
    return inspect_image_size_distribution(metadata, max_images=300)


def build_class_balance_chart(metadata):
    counts = (
        metadata.groupby(["split", "label"])
        .size()
        .reset_index(name="count")
        .assign(label=lambda df: df["label"].str.title())
    )
    fig = px.bar(
        counts,
        x="split",
        y="count",
        color="label",
        barmode="group",
        template="plotly_white",
        color_discrete_map={"Normal": "#1d4ed8", "Pneumonia": "#0f766e"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Class",
        xaxis_title="Dataset split",
        yaxis_title="Image count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_split_chart(split_counts):
    split_df = (
        split_counts.rename_axis("split")
        .reset_index(name="count")
        .assign(split=lambda df: df["split"].str.title())
    )
    fig = px.bar(
        split_df,
        x="split",
        y="count",
        template="plotly_white",
        color="split",
        color_discrete_sequence=["#2447B2", "#0f766e", "#94a3b8"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="Dataset split",
        yaxis_title="Image count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_sample_gallery(metadata) -> None:
    st.markdown("### Sample chest X-rays")
    st.write(
        "These example images make the dataset feel concrete and show the kind of scans the model is asked to separate into Normal and Pneumonia."
    )

    sample_df = get_selectable_example_images(
        preferred_split="test",
        labels=("NORMAL", "PNEUMONIA"),
        n_per_label=2,
    )

    gallery_columns = st.columns(max(len(sample_df), 1), gap="large")
    for column, row in zip(gallery_columns, sample_df.itertuples(index=False)):
        with column:
            image_path = resolve_project_path(row.image_path)
            with Image.open(image_path) as image:
                st.image(
                    image.copy(),
                    caption=f"{row.label.title()} | {getattr(row, 'source_split', row.split)}",
                    use_column_width=True,
                )
            st.caption(row.filename)


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Data & Context",
    title="Understanding the chest X-ray dataset before modeling",
    description=(
        "This project uses a public chest X-ray dataset to study a binary classification task: distinguishing pneumonia cases from normal images. "
        "The goal is not to simulate clinical diagnosis, but to explore how a computer vision workflow performs, how evaluation metrics should be interpreted, "
        "and how modeling choices affect outcomes."
    ),
    chips=["Binary classification", "Chest X-ray data", "Evaluation context", "Educational use only"],
)

try:
    metadata, overview = load_metadata()
except (FileNotFoundError, ValueError) as exc:
    metadata, overview = None, None
    render_surface_card(
        eyebrow="Setup Status",
        title="The dataset has not been downloaded yet",
        body=(
            "This page is waiting for the chest X-ray dataset to be available locally.\n"
            "Next step: run `python scripts/download_data.py` from the project root.\n"
            "Once the dataset is in place, the page will automatically show dataset summary cards, charts, and sample images."
        ),
    )
    st.caption(f"Current loader message: {exc}")

if metadata is not None and overview is not None:
    total_images = len(metadata)
    normal_images = int((metadata["label"] == "NORMAL").sum())
    pneumonia_images = int((metadata["label"] == "PNEUMONIA").sum())
    split_counts = metadata["split"].value_counts()

    top_metric_cols = st.columns(4, gap="large")
    top_metric_cols[0].metric("Total images", f"{total_images:,}")
    top_metric_cols[1].metric("Normal", f"{normal_images:,}")
    top_metric_cols[2].metric("Pneumonia", f"{pneumonia_images:,}")
    top_metric_cols[3].metric("Common size", overview["image_size"])

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    split_metric_cols = st.columns(3, gap="large")
    split_metric_cols[0].metric("Train images", f"{split_counts.get('train', 0):,}")
    split_metric_cols[1].metric("Validation images", f"{split_counts.get('val', 0):,}")
    split_metric_cols[2].metric("Test images", f"{split_counts.get('test', 0):,}")

    st.caption(
        f"Source folder: `{overview['dataset_root']}`. These cards establish the scale of the binary classification task and how the dataset is split."
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Class balance by split")
        st.plotly_chart(build_class_balance_chart(metadata), use_container_width=True)
        render_surface_card(
            eyebrow="Interpretation",
            title="How class balance shapes evaluation",
            body=(
                "What this shows:\n"
                "This chart summarizes how images are distributed across the Normal and Pneumonia classes.\n\n"
                "How to interpret it:\n"
                "Class balance matters because models can become biased toward the more common category, especially in binary classification tasks.\n\n"
                "Why it matters:\n"
                "Understanding the label distribution helps explain why metrics such as recall and precision can move differently, and why evaluation needs to go beyond overall accuracy."
            ),
        )
    with right:
        st.markdown("### Train / validation / test split")
        st.plotly_chart(build_split_chart(split_counts), use_container_width=True)
        render_surface_card(
            eyebrow="Interpretation",
            title="How the split structure supports evaluation",
            body=(
                "What this shows:\n"
                "This view breaks the dataset into training, validation, and test portions used throughout the project.\n\n"
                "How to interpret it:\n"
                "The training split is used to fit the model, the validation split supports model selection and monitoring, and the test split is held aside to measure final performance.\n\n"
                "Why it matters:\n"
                "A clear split structure helps make the evaluation more trustworthy and reduces the risk of over-interpreting results from a single stage of training."
            ),
        )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    render_surface_card(
        eyebrow="Responsible Interpretation",
        title="Educational analysis, not clinical guidance",
        body=(
            "The dataset and predictions shown in this project are for educational analysis only. "
            "This app is not intended for clinical deployment, diagnostic support, or treatment decision-making."
        ),
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    try:
        render_sample_gallery(metadata)
    except (ValueError, FileNotFoundError, OSError) as exc:
        render_surface_card(
            eyebrow="Samples",
            title="Example images are not available yet",
            body=(
                "This gallery prefers a few example scans from the downloaded dataset, with a focus on the test split. "
                "Once `python scripts/download_data.py` has been run successfully, the page will surface those examples automatically.\n"
                f"Current message: {exc}"
            ),
        )

    render_surface_card(
        eyebrow="Gallery Interpretation",
        title="How to read the example images",
        body=(
            "What this shows:\n"
            "These example chest X-rays give a visual sense of the images used in the dataset.\n\n"
            "How to interpret it:\n"
            "The gallery is not meant to teach diagnosis directly. Instead, it helps show the type of visual data the model is learning from and the variation that may exist across examples.\n\n"
            "Why it matters:\n"
            "Seeing real examples makes the task more concrete and helps users connect the prediction workflow to the underlying data."
        ),
    )

    st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.05, 0.95], gap="large")
    with bottom_left:
        render_surface_card(
            eyebrow="Metadata",
            title="What is tracked for each image",
            body=(
                "The loader builds a reusable metadata table with image path, split, label, and filename so the same structure "
                "can support exploration, training, evaluation, and demo experiences."
            ),
        )
    with bottom_right:
        try:
            size_profile = load_size_profile()
            render_surface_card(
                eyebrow="Image Sizes",
                title="Why resolution inspection helps",
                body=(
                    f"A quick scan of {len(size_profile):,} images helps confirm common dimensions and flags whether preprocessing "
                    "will need to standardize varied image sizes before model training."
                ),
            )
        except ValueError:
            render_surface_card(
                eyebrow="Image Sizes",
                title="Why resolution inspection helps",
                body=(
                    "A quick size distribution check is available when readable image files are present, helping validate preprocessing assumptions."
                ),
            )
