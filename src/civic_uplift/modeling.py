from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "district",
    "age_band",
    "race",
    "turnout_history",
    "gender",
    "education",
    "prior_turnout_rate",
    "donor_propensity",
    "digital_engagement",
    "contactability_score",
    "issue_salience_score",
]
CATEGORICAL_FEATURES = ["district", "age_band", "race", "turnout_history", "gender", "education"]
NUMERIC_FEATURES = [
    "prior_turnout_rate",
    "donor_propensity",
    "digital_engagement",
    "contactability_score",
    "issue_salience_score",
]


@dataclass
class ModelArtifacts:
    enriched: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    recommendation_summary: pd.DataFrame
    subgroup_uplift: pd.DataFrame


def _build_preprocessor() -> ColumnTransformer:
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )


def _fit_single_model(X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Pipeline:
    estimator = LogisticRegression(max_iter=2000) if model_name == "logistic_regression" else GradientBoostingClassifier(random_state=42)
    return Pipeline([("prep", _build_preprocessor()), ("model", estimator)])


def train_response_models(df: pd.DataFrame) -> ModelArtifacts:
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["action_taken"],
    )

    model_specs = {
        "logistic_regression": LogisticRegression(max_iter=2000, solver="liblinear"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    metrics_rows: List[dict] = []
    calibration_rows: List[dict] = []
    full_predictions = df.copy()

    for model_name, estimator in model_specs.items():
        pipeline = Pipeline([("prep", _build_preprocessor()), ("model", estimator)])
        pipeline.fit(train_df[FEATURES], train_df["action_taken"])
        preds = pipeline.predict_proba(test_df[FEATURES])[:, 1]
        metrics_rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(test_df["action_taken"], preds),
                "average_precision": average_precision_score(test_df["action_taken"], preds),
                "brier_score": brier_score_loss(test_df["action_taken"], preds),
            }
        )

        frac_pos, mean_pred = calibration_curve(test_df["action_taken"], preds, n_bins=10, strategy="quantile")
        calibration_rows.extend(
            {
                "model": model_name,
                "mean_predicted_probability": x,
                "fraction_positive": y,
            }
            for x, y in zip(mean_pred, frac_pos)
        )
        full_predictions[f"score_{model_name}"] = pipeline.predict_proba(df[FEATURES])[:, 1]

    uplift_predictions = estimate_message_uplift(df)
    enriched = full_predictions.merge(uplift_predictions, on="voter_id", how="left")
    recommendation_summary = build_recommendation_summary(enriched)
    subgroup_uplift = build_subgroup_uplift(enriched)
    return ModelArtifacts(
        enriched=enriched,
        metrics=pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False),
        calibration=pd.DataFrame(calibration_rows),
        recommendation_summary=recommendation_summary,
        subgroup_uplift=subgroup_uplift,
    )


def estimate_message_uplift(df: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.DataFrame({"voter_id": df["voter_id"]})
    treatment_scores: Dict[str, np.ndarray] = {}

    for treatment in ["control", "economic_security", "community_voice", "future_forward"]:
        binary_df = df.copy()
        binary_df["is_treatment"] = binary_df["treatment"].eq(treatment).astype(int)
        feature_cols = FEATURES + ["is_treatment"]
        model = Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            (
                                "categorical",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                    ]
                                ),
                                CATEGORICAL_FEATURES,
                            ),
                            (
                                "numeric",
                                Pipeline(
                                    [
                                        ("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler()),
                                    ]
                                ),
                                NUMERIC_FEATURES + ["is_treatment"],
                            ),
                        ]
                    ),
                ),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        )
        model.fit(binary_df[feature_cols], binary_df["action_taken"])

        scenario = df[FEATURES].copy()
        scenario["is_treatment"] = 1
        treated_pred = model.predict_proba(scenario)[:, 1]
        scenario["is_treatment"] = 0
        control_pred = model.predict_proba(scenario)[:, 1]
        treatment_scores[treatment] = treated_pred if treatment == "control" else np.clip(treated_pred - control_pred, -1, 1)

    for treatment, scores in treatment_scores.items():
        col = "pred_control_probability" if treatment == "control" else f"uplift_{treatment}"
        predictions[col] = scores

    uplift_cols = [col for col in predictions.columns if col.startswith("uplift_")]
    predictions["best_message"] = predictions[uplift_cols].idxmax(axis=1).str.replace("uplift_", "", regex=False)
    predictions["best_uplift"] = predictions[uplift_cols].max(axis=1)
    return predictions


def build_recommendation_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    summary = (
        enriched.groupby("best_message", as_index=False)
        .agg(
            voters=("voter_id", "count"),
            avg_uplift=("best_uplift", "mean"),
            avg_response_score=("score_gradient_boosting", "mean"),
        )
        .sort_values(["avg_uplift", "voters"], ascending=[False, False])
    )
    return summary


def build_subgroup_uplift(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    uplift_columns = {
        "economic_security": "uplift_economic_security",
        "community_voice": "uplift_community_voice",
        "future_forward": "uplift_future_forward",
    }
    for dimension in ["district", "age_band", "race", "turnout_history"]:
        grouped = enriched.groupby(dimension)
        for subgroup, frame in grouped:
            for message, col in uplift_columns.items():
                rows.append(
                    {
                        "dimension": dimension,
                        "subgroup": subgroup,
                        "message": message,
                        "avg_uplift": frame[col].mean(),
                        "sample_size": len(frame),
                    }
                )
    return pd.DataFrame(rows)
