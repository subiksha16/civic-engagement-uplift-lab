from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .data import build_feature_table, generate_synthetic_voter_data, get_paths, save_dataset
from .modeling import ModelArtifacts, train_response_models


def compute_experiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("voters", df)
    summary = con.execute(
        """
        with message_rates as (
            select
                treatment,
                message_label,
                count(*) as audience_size,
                avg(action_taken) as action_rate,
                avg(donated) as donation_rate,
                avg(pledged_to_vote) as pledge_rate
            from voters
            group by 1, 2
        ),
        control_rate as (
            select avg(action_taken) as control_action_rate
            from voters
            where treatment = 'control'
        )
        select
            m.*,
            m.action_rate - c.control_action_rate as ate_vs_control
        from message_rates m
        cross join control_rate c
        order by ate_vs_control desc, audience_size desc
        """
    ).df()
    con.close()
    return summary


def compute_top_segments(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in enriched.iterrows():
        rows.append(
            {
                "district": row["district"],
                "age_band": row["age_band"],
                "turnout_history": row["turnout_history"],
                "best_message": row["best_message"],
                "best_uplift": row["best_uplift"],
                "response_score": row["score_gradient_boosting"],
            }
        )
    segment_df = pd.DataFrame(rows)
    top_segments = (
        segment_df.groupby(["district", "age_band", "turnout_history", "best_message"], as_index=False)
        .agg(
            avg_uplift=("best_uplift", "mean"),
            avg_response_score=("response_score", "mean"),
            audience=("best_uplift", "size"),
        )
        .query("audience >= 80")
        .sort_values(["avg_uplift", "avg_response_score"], ascending=False)
        .head(15)
    )
    return top_segments


def write_artifacts(
    feature_df: pd.DataFrame,
    experiment_summary: pd.DataFrame,
    model_artifacts: ModelArtifacts,
    top_segments: pd.DataFrame,
) -> None:
    paths = get_paths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(paths.data_dir / "feature_table.parquet", index=False)
    experiment_summary.to_csv(paths.data_dir / "experiment_summary.csv", index=False)
    model_artifacts.metrics.to_csv(paths.data_dir / "model_metrics.csv", index=False)
    model_artifacts.calibration.to_csv(paths.data_dir / "calibration_curve.csv", index=False)
    model_artifacts.enriched.to_parquet(paths.data_dir / "scored_voters.parquet", index=False)
    model_artifacts.recommendation_summary.to_csv(paths.data_dir / "message_recommendations.csv", index=False)
    model_artifacts.subgroup_uplift.to_csv(paths.data_dir / "subgroup_uplift.csv", index=False)
    top_segments.to_csv(paths.data_dir / "top_segments.csv", index=False)


def run_pipeline(seed: int = 42) -> dict:
    paths = get_paths()
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    raw_df = generate_synthetic_voter_data(seed=seed)
    save_dataset(raw_df)
    feature_df = build_feature_table(raw_df)
    experiment_summary = compute_experiment_summary(feature_df)
    model_artifacts = train_response_models(feature_df)
    top_segments = compute_top_segments(model_artifacts.enriched)
    write_artifacts(feature_df, experiment_summary, model_artifacts, top_segments)
    return {
        "raw_df": raw_df,
        "feature_df": feature_df,
        "experiment_summary": experiment_summary,
        "model_artifacts": model_artifacts,
        "top_segments": top_segments,
        "root": Path(paths.root),
    }
