from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civic_uplift.analysis import run_pipeline


def test_pipeline_outputs_expected_shapes() -> None:
    artifacts = run_pipeline(seed=7)
    assert len(artifacts["feature_df"]) == 12000
    assert not artifacts["experiment_summary"].empty
    assert not artifacts["model_artifacts"].metrics.empty
    assert {"uplift_economic_security", "uplift_community_voice", "uplift_future_forward"}.issubset(
        set(artifacts["model_artifacts"].enriched.columns)
    )
