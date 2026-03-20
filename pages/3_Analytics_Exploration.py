from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.layout import render_page_header, render_sidebar, render_surface_card, setup_page
from src.load_data import build_metadata_dataframe, inspect_image_size_distribution


@st.cache_data(show_spinner=False)
def load_metadata():
    return build_metadata_dataframe()


@st.cache_data(show_spinner=False)
def load_size_metadata():
    metadata = load_metadata()
    return inspect_image_size_distribution(metadata, max_images=500)


def build_split_balance_chart(metadata):
    chart_df = (
        metadata.groupby(["split", "label"])
        .size()
        .reset_index(name="count")
        .assign(label=lambda df: df["label"].str.title())
    )
    fig = px.bar(
        chart_df,
        x="split",
        y="count",
        color="label",
        barmode="group",
        template="plotly_white",
        color_discrete_map={"Normal": "#1d4ed8", "Pneumonia": "#0f766e"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="Class",
        xaxis_title="Dataset split",
        yaxis_title="Image count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_class_proportion_chart(metadata):
    chart_df = (
        metadata.groupby("label")
        .size()
        .reset_index(name="count")
        .assign(label=lambda df: df["label"].str.title())
    )
    fig = px.pie(
        chart_df,
        names="label",
        values="count",
        hole=0.56,
        template="plotly_white",
        color="label",
        color_discrete_map={"Normal": "#1d4ed8", "Pneumonia": "#0f766e"},
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_dimension_chart(size_df):
    chart_df = size_df.copy()
    chart_df["label"] = chart_df["label"].str.title()
    fig = px.scatter(
        chart_df,
        x="width",
        y="height",
        color="label",
        facet_col="split",
        template="plotly_white",
        opacity=0.7,
        color_discrete_map={"Normal": "#1d4ed8", "Pneumonia": "#0f766e"},
        hover_data=["filename"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="Class",
        xaxis_title="Width (px)",
        yaxis_title="Height (px)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title()))
    return fig


setup_page()
render_sidebar()

render_page_header(
    eyebrow="Analytics Exploration",
    title="Interactive exploration of the chest X-ray dataset",
    description=(
        "This dashboard is a quick way to inspect class composition, split-level balance, and image-size patterns "
        "before moving into training or evaluation. The goal is to make the dataset legible at a glance."
    ),
    chips=["Interactive filters", "Plotly charts", "Dataset diagnostics", "Dashboard-style exploration"],
)

try:
    metadata = load_metadata()
except (FileNotFoundError, ValueError) as exc:
    metadata = None
    st.error(str(exc))
    st.info(
        "Run `python scripts/download_data.py` and make sure `data/raw/` contains the expected chest X-ray folders before using this dashboard."
    )

if metadata is not None:
    available_splits = sorted(metadata["split"].unique().tolist())
    available_labels = sorted(metadata["label"].unique().tolist())

    filter_left, filter_right, summary_col = st.columns([1.05, 1.05, 0.9], gap="large")
    with filter_left:
        selected_splits = st.multiselect(
            "Filter by split",
            options=available_splits,
            default=available_splits,
            help="Limit the dashboard to one or more dataset partitions.",
        )
    with filter_right:
        selected_labels = st.multiselect(
            "Filter by label",
            options=available_labels,
            default=available_labels,
            help="Focus the charts on one or both target classes.",
        )
    with summary_col:
        render_surface_card(
            eyebrow="Filtered View",
            title="Current selection",
            body=(
                f"{len(selected_splits)} split(s) selected\n"
                f"{len(selected_labels)} class label(s) selected\n"
                f"{len(metadata):,} total rows available before filtering"
            ),
        )

    filtered_metadata = metadata[
        metadata["split"].isin(selected_splits) & metadata["label"].isin(selected_labels)
    ].copy()

    if filtered_metadata.empty:
        st.warning("No rows match the current filter combination. Adjust the split or label filters to continue.")
    else:
        metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
        metric_col1.metric("Filtered images", f"{len(filtered_metadata):,}")
        metric_col2.metric("Filtered splits", ", ".join(selected_splits))
        metric_col3.metric("Filtered labels", ", ".join(label.title() for label in selected_labels))

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        top_left, top_right = st.columns([1.15, 0.85], gap="large")
        with top_left:
            st.markdown("### Class balance by split")
            st.plotly_chart(build_split_balance_chart(filtered_metadata), use_container_width=True)
            st.markdown(
                "What this chart shows: The grouped bars compare how many Normal and Pneumonia images appear in each selected split. "
                "This helps reveal whether the training and evaluation partitions are evenly structured or visibly skewed."
            )
        with top_right:
            st.markdown("### Overall class proportions")
            st.plotly_chart(build_class_proportion_chart(filtered_metadata), use_container_width=True)
            st.markdown(
                "What this chart shows: The donut chart summarizes the class mix in the filtered subset. "
                "It gives a quick read on whether the current slice of data is balanced or dominated by one label."
            )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        st.markdown("### Image width and height distribution")
        try:
            size_metadata = load_size_metadata()
            filtered_size_metadata = size_metadata[
                size_metadata["split"].isin(selected_splits) & size_metadata["label"].isin(selected_labels)
            ].copy()

            if filtered_size_metadata.empty:
                st.info("No image dimension rows matched the current filter selection.")
            else:
                st.plotly_chart(build_dimension_chart(filtered_size_metadata), use_container_width=True)
                st.markdown(
                    "What this chart shows: Each point represents one readable image, plotted by width and height. "
                    "This helps confirm whether dimensions are consistent across splits or whether preprocessing will need to account for variability."
                )
        except ValueError as exc:
            render_surface_card(
                eyebrow="Dimensions Not Available",
                title="Image-size profiling will appear here when readable files are available",
                body=(
                    "The dashboard could not build image width and height metadata yet. "
                    f"Reason: {exc}"
                ),
            )

        st.markdown("<div class='section-spacer'></div>", unsafe_allow_html=True)

        insight_left, insight_right = st.columns([1.1, 0.9], gap="large")
        with insight_left:
            render_surface_card(
                eyebrow="Why This Matters",
                title="Exploration shapes the rest of the project",
                body=(
                    "Early analytics help establish whether the dataset is balanced, whether the splits look trustworthy, "
                    "and whether the raw images will need standardized preprocessing before any modeling work begins."
                ),
            )
        with insight_right:
            render_surface_card(
                eyebrow="Dashboard Intent",
                title="Built for fast review",
                body=(
                    "This page is intentionally lightweight and visual so users can understand the data profile "
                    "in a few minutes without reading code or opening a notebook."
                ),
            )
