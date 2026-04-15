from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civic_uplift.analysis import run_pipeline
from civic_uplift.data import MESSAGE_LABELS


st.set_page_config(
    page_title="Civic Engagement Uplift Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict:
    data_dir = ROOT / "data" / "processed"
    scored_path = data_dir / "scored_voters.parquet"
    if not scored_path.exists():
        return run_pipeline()
    return {
        "feature_df": pd.read_parquet(data_dir / "feature_table.parquet"),
        "experiment_summary": pd.read_csv(data_dir / "experiment_summary.csv"),
        "model_metrics": pd.read_csv(data_dir / "model_metrics.csv"),
        "calibration": pd.read_csv(data_dir / "calibration_curve.csv"),
        "scored_voters": pd.read_parquet(scored_path),
        "subgroup_uplift": pd.read_csv(data_dir / "subgroup_uplift.csv"),
        "top_segments": pd.read_csv(data_dir / "top_segments.csv"),
        "message_recommendations": pd.read_csv(data_dir / "message_recommendations.csv"),
    }


artifacts = load_artifacts()
scored_voters = artifacts.get("scored_voters", artifacts["model_artifacts"].enriched if "model_artifacts" in artifacts else None)
experiment_summary = artifacts.get("experiment_summary")
subgroup_uplift = artifacts.get("subgroup_uplift", artifacts["model_artifacts"].subgroup_uplift if "model_artifacts" in artifacts else None)
model_metrics = artifacts.get("model_metrics", artifacts["model_artifacts"].metrics if "model_artifacts" in artifacts else None)
calibration = artifacts.get("calibration", artifacts["model_artifacts"].calibration if "model_artifacts" in artifacts else None)
top_segments = artifacts.get("top_segments")

st.title("Civic Engagement Uplift Lab")
st.caption("A campaign analytics sandbox for message testing, persuasion modeling, and segment-level treatment recommendations.")

with st.sidebar:
    st.header("Audience Filters")
    selected_districts = st.multiselect("District", sorted(scored_voters["district"].unique()), default=sorted(scored_voters["district"].unique()))
    selected_age = st.multiselect("Age band", sorted(scored_voters["age_band"].unique()), default=sorted(scored_voters["age_band"].unique()))
    selected_turnout = st.multiselect(
        "Turnout history",
        ["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
    )
    min_contact = st.slider("Minimum contactability", min_value=10, max_value=99, value=20)

filtered = scored_voters[
    scored_voters["district"].isin(selected_districts)
    & scored_voters["age_band"].isin(selected_age)
    & scored_voters["turnout_history"].isin(selected_turnout)
    & scored_voters["contactability_score"].ge(min_contact)
].copy()

if filtered.empty:
    st.warning("No voters match the current filter set. Loosen the filters to see recommendations.")
    st.stop()

best_message_share = (
    filtered["best_message"]
    .map(MESSAGE_LABELS)
    .value_counts(normalize=True)
    .rename_axis("message")
    .reset_index(name="share")
)
recommended_message = best_message_share.iloc[0]["message"]
median_uplift = filtered["best_uplift"].mean()
response_rate = filtered["score_gradient_boosting"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Audience size", f"{len(filtered):,}")
kpi2.metric("Recommended message", recommended_message)
kpi3.metric("Predicted uplift", f"{median_uplift:.1%}")
kpi4.metric("Predicted response rate", f"{response_rate:.1%}")

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Experiment Readout")
    fig = px.bar(
        experiment_summary.sort_values("ate_vs_control", ascending=False),
        x="message_label",
        y="ate_vs_control",
        color="message_label",
        text=experiment_summary.sort_values("ate_vs_control", ascending=False)["ate_vs_control"].map(lambda x: f"{x:.1%}"),
        color_discrete_sequence=["#0f766e", "#7aa874", "#e9b949", "#b8c4ce"],
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="#f6f7f3", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Message Mix in Filtered Audience")
    donut = px.pie(best_message_share, names="message", values="share", hole=0.56, color="message", color_discrete_sequence=["#0f766e", "#7aa874", "#e9b949"])
    donut.update_layout(showlegend=True, paper_bgcolor="#f6f7f3")
    donut.update_traces(textinfo="percent+label")
    st.plotly_chart(donut, use_container_width=True)

with right:
    st.subheader("Uplift by Age Segment")
    heatmap_df = subgroup_uplift.query("dimension == 'age_band'").pivot(index="subgroup", columns="message", values="avg_uplift").reset_index()
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_df[[c for c in heatmap_df.columns if c != "subgroup"]].values,
            x=[c.replace("_", " ").title() for c in heatmap_df.columns if c != "subgroup"],
            y=heatmap_df["subgroup"],
            colorscale="YlGnBu",
            text=[[f"{value:.1%}" for value in row] for row in heatmap_df[[c for c in heatmap_df.columns if c != "subgroup"]].values],
            texttemplate="%{text}",
        )
    )
    heatmap.update_layout(paper_bgcolor="#f6f7f3")
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Model Diagnostics")
    metrics_table = model_metrics.copy()
    for col in ["roc_auc", "average_precision", "brier_score"]:
        metrics_table[col] = metrics_table[col].map(lambda x: f"{x:.3f}")
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

st.subheader("Top Persuadable Segments")
segment_view = top_segments.copy()
segment_view["avg_uplift"] = segment_view["avg_uplift"].map(lambda x: f"{x:.1%}")
segment_view["avg_response_score"] = segment_view["avg_response_score"].map(lambda x: f"{x:.1%}")
segment_view["best_message"] = segment_view["best_message"].map(MESSAGE_LABELS)
st.dataframe(segment_view.head(10), use_container_width=True, hide_index=True)

st.subheader("Calibration Curve")
cal_fig = px.line(
    calibration,
    x="mean_predicted_probability",
    y="fraction_positive",
    color="model",
    markers=True,
    color_discrete_sequence=["#0f766e", "#e9b949"],
)
cal_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#102a43"})
cal_fig.update_layout(plot_bgcolor="white", paper_bgcolor="#f6f7f3", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
st.plotly_chart(cal_fig, use_container_width=True)
