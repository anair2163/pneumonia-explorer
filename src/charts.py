from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PRIMARY_BLUE = "#1d4ed8"
PRIMARY_TEAL = "#0f766e"
ACCENT_AMBER = "#d97706"
SURFACE_LINE = "rgba(15, 23, 42, 0.08)"
GRID_COLOR = "rgba(148, 163, 184, 0.22)"
TEXT_COLOR = "#0f172a"
MUTED_COLOR = "#475569"


def build_class_balance_chart(
    data: pd.DataFrame,
    *,
    split_col: str = "split",
    label_col: str = "label",
    count_col: str | None = None,
    title: str = "Class Balance by Split",
):
    chart_df = _ensure_count_dataframe(
        data,
        group_cols=[split_col, label_col],
        count_col=count_col,
    ).copy()
    chart_df[label_col] = chart_df[label_col].astype(str).str.title()

    fig = px.bar(
        chart_df,
        x=split_col,
        y="count",
        color=label_col,
        barmode="group",
        template="plotly_white",
        color_discrete_map={"Normal": PRIMARY_BLUE, "Pneumonia": PRIMARY_TEAL},
        title=title,
    )
    return _apply_base_layout(
        fig,
        xaxis_title="Dataset split",
        yaxis_title="Image count",
        legend_title="Class",
    )


def build_confusion_matrix_heatmap(
    confusion_values,
    *,
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
):
    labels = labels or ["NORMAL", "PNEUMONIA"]
    matrix_df = pd.DataFrame(confusion_values, index=labels, columns=labels)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=matrix_df.columns.tolist(),
            y=matrix_df.index.tolist(),
            colorscale=[
                [0.0, "#eff6ff"],
                [0.35, "#93c5fd"],
                [0.7, "#2563eb"],
                [1.0, "#0f766e"],
            ],
            showscale=True,
            colorbar=dict(title="Count"),
            text=matrix_df.values,
            texttemplate="%{text}",
            textfont=dict(color=TEXT_COLOR, size=14),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )
    fig.update_xaxes(side="top")
    return fig


def build_roc_curve(
    roc_df: pd.DataFrame,
    *,
    title: str = "ROC Curve",
    auc_value: float | None = None,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_df["false_positive_rate"],
            y=roc_df["true_positive_rate"],
            mode="lines",
            line=dict(color=PRIMARY_TEAL, width=3),
            name="Model ROC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=MUTED_COLOR, dash="dash"),
            name="Chance",
        )
    )

    title_text = title
    if auc_value is not None:
        title_text = f"{title} (AUC = {auc_value:.3f})"

    fig.update_layout(
        title=title_text,
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def build_probability_distribution_histogram(
    predictions_df: pd.DataFrame,
    *,
    probability_col: str = "prob_pneumonia",
    color_col: str = "true_label",
    title: str = "Prediction Probability Distribution",
    nbins: int = 30,
):
    chart_df = predictions_df.copy()
    chart_df[color_col] = chart_df[color_col].astype(str).str.title()

    fig = px.histogram(
        chart_df,
        x=probability_col,
        color=color_col,
        nbins=nbins,
        barmode="overlay",
        opacity=0.7,
        template="plotly_white",
        color_discrete_map={"Normal": PRIMARY_BLUE, "Pneumonia": PRIMARY_TEAL},
        title=title,
    )
    return _apply_base_layout(
        fig,
        xaxis_title="Predicted probability for Pneumonia",
        yaxis_title="Image count",
        legend_title="True label",
    )


def build_metric_comparison_chart(
    comparison_df: pd.DataFrame,
    *,
    model_col: str = "model",
    metric_cols: list[str] | None = None,
    title: str = "Metric Comparison",
):
    metric_cols = metric_cols or ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [metric for metric in metric_cols if metric in comparison_df.columns]
    chart_df = comparison_df[[model_col] + available_metrics].melt(
        id_vars=model_col,
        value_vars=available_metrics,
        var_name="metric",
        value_name="score",
    )
    chart_df = chart_df.dropna(subset=["score"])
    chart_df["metric"] = chart_df["metric"].str.upper()

    fig = px.bar(
        chart_df,
        x="metric",
        y="score",
        color=model_col,
        barmode="group",
        template="plotly_white",
        color_discrete_sequence=[PRIMARY_BLUE, PRIMARY_TEAL, ACCENT_AMBER],
        title=title,
    )
    fig.update_yaxes(range=[0, 1])
    return _apply_base_layout(
        fig,
        xaxis_title="Metric",
        yaxis_title="Score",
        legend_title="Model",
    )


def build_metric_trend_chart(
    data: pd.DataFrame | None = None,
    *,
    x_col: str = "epoch",
    y_col: str = "validation_f1",
    title: str = "Validation Metric by Epoch",
):
    if data is None:
        data = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4, 5],
                "validation_f1": [0.71, 0.79, 0.84, 0.89, 0.92],
            }
        )

    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        markers=True,
        template="plotly_white",
        color_discrete_sequence=[ACCENT_AMBER],
        title=title,
    )
    return _apply_base_layout(
        fig,
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        legend_title="",
    )


def _ensure_count_dataframe(
    data: pd.DataFrame,
    *,
    group_cols: list[str],
    count_col: str | None,
) -> pd.DataFrame:
    if count_col is not None and count_col in data.columns:
        return data.rename(columns={count_col: "count"})
    return data.groupby(group_cols).size().reset_index(name="count")


def _apply_base_layout(
    fig,
    *,
    xaxis_title: str,
    yaxis_title: str,
    legend_title: str,
):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        legend=dict(
            title=legend_title,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        bargap=0.18,
    )
    fig.update_xaxes(
        title=xaxis_title,
        showgrid=False,
        zeroline=False,
        linecolor=SURFACE_LINE,
    )
    fig.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=SURFACE_LINE,
    )
    return fig
