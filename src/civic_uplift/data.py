from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


MESSAGE_LABELS = {
    "control": "Control",
    "economic_security": "Economic Security",
    "community_voice": "Community Voice",
    "future_forward": "Future Forward",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def get_paths() -> ProjectPaths:
    return ProjectPaths(root=Path(__file__).resolve().parents[2])


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_synthetic_voter_data(n_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    districts = np.array(["AZ-01", "GA-06", "MI-08", "NC-13", "PA-07", "TX-32", "WI-03"])
    district_weights = np.array([0.12, 0.13, 0.15, 0.17, 0.15, 0.14, 0.14])
    races = np.array(["White", "Black", "Latino", "Asian", "Multi/Other"])
    race_weights = np.array([0.44, 0.20, 0.22, 0.08, 0.06])
    age_bands = np.array(["18-29", "30-44", "45-64", "65+"])
    age_weights = np.array([0.26, 0.31, 0.29, 0.14])
    turnout_segments = np.array(["Low", "Medium", "High"])
    turnout_weights = np.array([0.32, 0.44, 0.24])
    treatments = np.array(list(MESSAGE_LABELS.keys()))

    df = pd.DataFrame(
        {
            "voter_id": np.arange(1, n_rows + 1),
            "district": rng.choice(districts, size=n_rows, p=district_weights),
            "age_band": rng.choice(age_bands, size=n_rows, p=age_weights),
            "race": rng.choice(races, size=n_rows, p=race_weights),
            "turnout_history": rng.choice(turnout_segments, size=n_rows, p=turnout_weights),
            "gender": rng.choice(["Female", "Male"], size=n_rows, p=[0.53, 0.47]),
            "education": rng.choice(
                ["High school", "Some college", "College+", "Graduate"],
                size=n_rows,
                p=[0.25, 0.30, 0.28, 0.17],
            ),
            "treatment": rng.choice(treatments, size=n_rows, p=[0.24, 0.26, 0.25, 0.25]),
            "prior_turnout_rate": np.clip(rng.normal(0.56, 0.20, size=n_rows), 0.02, 0.98),
            "donor_propensity": np.clip(rng.beta(2.1, 5.5, size=n_rows), 0.01, 0.99),
            "digital_engagement": np.clip(rng.beta(2.7, 2.9, size=n_rows), 0.01, 0.99),
            "contactability_score": np.clip(rng.normal(58, 16, size=n_rows), 10, 99),
            "issue_salience_score": np.clip(rng.normal(61, 15, size=n_rows), 10, 99),
        }
    )

    district_competitiveness = {
        "AZ-01": 0.76,
        "GA-06": 0.71,
        "MI-08": 0.79,
        "NC-13": 0.73,
        "PA-07": 0.84,
        "TX-32": 0.69,
        "WI-03": 0.82,
    }
    district_digital = {
        "AZ-01": 0.03,
        "GA-06": 0.05,
        "MI-08": 0.02,
        "NC-13": 0.01,
        "PA-07": 0.04,
        "TX-32": 0.02,
        "WI-03": -0.01,
    }

    age_effect = {"18-29": -0.24, "30-44": 0.03, "45-64": 0.10, "65+": 0.14}
    race_effect = {"White": 0.00, "Black": 0.08, "Latino": 0.05, "Asian": 0.02, "Multi/Other": 0.01}
    turnout_effect = {"Low": -0.30, "Medium": 0.02, "High": 0.19}

    base_logit = (
        -2.15
        + 1.05 * df["prior_turnout_rate"]
        + 0.85 * df["digital_engagement"]
        + 0.55 * df["donor_propensity"]
        + 0.014 * (df["contactability_score"] - 50)
        + 0.011 * (df["issue_salience_score"] - 50)
        + df["district"].map(district_competitiveness)
        + df["district"].map(district_digital)
        + df["age_band"].map(age_effect)
        + df["race"].map(race_effect)
        + df["turnout_history"].map(turnout_effect)
        + np.where(df["gender"].eq("Female"), 0.05, 0.00)
        + np.where(df["education"].eq("Graduate"), 0.09, 0.00)
        + np.where(df["education"].eq("College+"), 0.05, 0.00)
    )

    econ_uplift = (
        0.08
        + 0.32 * df["age_band"].isin(["30-44", "45-64"]).astype(float)
        + 0.18 * df["race"].isin(["Latino", "Black"]).astype(float)
        + 0.20 * df["turnout_history"].eq("Low").astype(float)
        + 0.14 * (df["issue_salience_score"] > 58).astype(float)
    )
    community_uplift = (
        0.05
        + 0.27 * df["race"].isin(["Black", "Latino"]).astype(float)
        + 0.18 * df["district"].isin(["GA-06", "PA-07", "WI-03"]).astype(float)
        + 0.11 * df["turnout_history"].eq("Medium").astype(float)
        + 0.15 * (df["contactability_score"] > 60).astype(float)
    )
    future_uplift = (
        0.06
        + 0.31 * df["age_band"].eq("18-29").astype(float)
        + 0.16 * df["digital_engagement"].gt(0.65).astype(float)
        + 0.12 * df["district"].isin(["AZ-01", "TX-32"]).astype(float)
        + 0.10 * df["turnout_history"].eq("Low").astype(float)
    )

    treatment_effect = np.select(
        [
            df["treatment"].eq("economic_security"),
            df["treatment"].eq("community_voice"),
            df["treatment"].eq("future_forward"),
        ],
        [econ_uplift, community_uplift, future_uplift],
        default=0.0,
    )
    treatment_effect += rng.normal(0, 0.07, size=n_rows)

    action_prob = sigmoid(base_logit + treatment_effect)
    df["action_taken"] = rng.binomial(1, action_prob)

    donation_prob = sigmoid(
        -3.0
        + 1.6 * df["donor_propensity"]
        + 0.7 * df["action_taken"]
        + 0.2 * df["treatment"].eq("economic_security").astype(float)
        + 0.15 * df["turnout_history"].eq("High").astype(float)
    )
    df["donated"] = rng.binomial(1, donation_prob)
    df["pledged_to_vote"] = np.where(df["action_taken"].eq(1), rng.binomial(1, 0.72, size=n_rows), 0)
    df["clicked_message"] = np.where(df["action_taken"].eq(1), rng.binomial(1, 0.81, size=n_rows), rng.binomial(1, 0.16, size=n_rows))
    df["message_label"] = df["treatment"].map(MESSAGE_LABELS)
    df["age_order"] = df["age_band"].map({"18-29": 0, "30-44": 1, "45-64": 2, "65+": 3})
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("voters", df)
    feature_df = con.execute(
        """
        with scored as (
            select
                voter_id,
                district,
                age_band,
                race,
                turnout_history,
                gender,
                education,
                treatment,
                message_label,
                prior_turnout_rate,
                donor_propensity,
                digital_engagement,
                contactability_score,
                issue_salience_score,
                action_taken,
                donated,
                pledged_to_vote,
                clicked_message,
                case
                    when prior_turnout_rate < 0.35 then 'Low history'
                    when prior_turnout_rate < 0.70 then 'Persuadable middle'
                    else 'Reliable voter'
                end as voter_profile,
                case
                    when contactability_score >= 72 then 'High'
                    when contactability_score >= 52 then 'Medium'
                    else 'Low'
                end as contactability_band
            from voters
        )
        select * from scored
        """
    ).df()
    con.close()
    return feature_df


def save_dataset(df: pd.DataFrame) -> Path:
    paths = get_paths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = paths.data_dir / "synthetic_voter_engagement.csv"
    df.to_csv(dataset_path, index=False)
    return dataset_path
