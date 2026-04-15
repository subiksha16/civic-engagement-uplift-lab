from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .data import MESSAGE_LABELS, get_paths


BRAND = {
    "ink": "#102a43",
    "teal": "#0f766e",
    "sage": "#7aa874",
    "gold": "#e9b949",
    "rose": "#d64562",
    "sand": "#f6f7f3",
}


def _setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = BRAND["sand"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#d9e2ec"
    plt.rcParams["axes.labelcolor"] = BRAND["ink"]
    plt.rcParams["text.color"] = BRAND["ink"]
    plt.rcParams["axes.titleweight"] = "bold"


def export_figures(experiment_summary: pd.DataFrame, subgroup_uplift: pd.DataFrame, metrics: pd.DataFrame, calibration: pd.DataFrame, top_segments: pd.DataFrame) -> list[Path]:
    _setup_theme()
    paths = get_paths()
    output_paths = [
        _save_experiment_chart(experiment_summary, paths.figures_dir),
        _save_heatmap(subgroup_uplift, paths.figures_dir),
        _save_metrics_chart(metrics, calibration, paths.figures_dir),
        _save_top_segments_chart(top_segments, paths.figures_dir),
    ]
    plt.close("all")
    return output_paths


def _save_experiment_chart(experiment_summary: pd.DataFrame, out_dir: Path) -> Path:
    order = experiment_summary.sort_values("ate_vs_control", ascending=False)["message_label"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=experiment_summary,
        x="message_label",
        y="ate_vs_control",
        order=order,
        hue="message_label",
        palette=[BRAND["teal"], BRAND["sage"], BRAND["gold"], "#b8c4ce"],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color=BRAND["ink"], linewidth=1.2, linestyle="--")
    ax.set_title("Average Treatment Effect vs. Control")
    ax.set_xlabel("")
    ax.set_ylabel("Lift in action rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    for idx, value in enumerate(experiment_summary.set_index("message_label").loc[order, "ate_vs_control"]):
        ax.text(idx, value + 0.003, f"{value:.1%}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = out_dir / "01_message_uplift.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return path


def _save_heatmap(subgroup_uplift: pd.DataFrame, out_dir: Path) -> Path:
    focus = subgroup_uplift.query("dimension == 'age_band'").copy()
    pivot = focus.pivot(index="subgroup", columns="message", values="avg_uplift")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Predicted uplift"}, ax=ax)
    ax.set_title("Predicted Uplift by Age Segment")
    ax.set_xlabel("Message")
    ax.set_ylabel("")
    fig.tight_layout()
    path = out_dir / "02_age_segment_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return path


def _save_metrics_chart(metrics: pd.DataFrame, calibration: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_melted = metrics.melt(id_vars="model", value_vars=["roc_auc", "average_precision", "brier_score"], var_name="metric", value_name="score")
    sns.barplot(data=metrics_melted, x="metric", y="score", hue="model", palette=[BRAND["teal"], BRAND["gold"]], ax=axes[0])
    axes[0].set_title("Model Quality")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")
    axes[0].legend(title="")

    for model_name, frame in calibration.groupby("model"):
        axes[1].plot(frame["mean_predicted_probability"], frame["fraction_positive"], marker="o", label=model_name.replace("_", " ").title())
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=BRAND["ink"], linewidth=1)
    axes[1].set_title("Calibration Curve")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed action rate")
    axes[1].legend(title="")
    fig.tight_layout()
    path = out_dir / "03_model_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return path


def _save_top_segments_chart(top_segments: pd.DataFrame, out_dir: Path) -> Path:
    chart_df = top_segments.head(10).copy()
    chart_df["segment"] = chart_df["district"] + " | " + chart_df["age_band"] + " | " + chart_df["turnout_history"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=chart_df,
        x="avg_response_score",
        y="avg_uplift",
        size="audience",
        hue="best_message",
        palette={
            "economic_security": BRAND["teal"],
            "community_voice": BRAND["sage"],
            "future_forward": BRAND["gold"],
        },
        sizes=(80, 600),
        ax=ax,
    )
    ax.set_title("Highest-Leverage Segments")
    ax.set_xlabel("Predicted response probability")
    ax.set_ylabel("Predicted uplift")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    for _, row in chart_df.iterrows():
        ax.text(row["avg_response_score"] + 0.003, row["avg_uplift"] + 0.001, row["district"], fontsize=9)
    ax.legend(title="")
    fig.tight_layout()
    path = out_dir / "04_top_segments.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    return path
