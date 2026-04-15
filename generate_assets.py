from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from civic_uplift.analysis import run_pipeline
from civic_uplift.visuals import export_figures


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    (ROOT / ".mplconfig").mkdir(exist_ok=True)
    artifacts = run_pipeline()
    figure_paths = export_figures(
        artifacts["experiment_summary"],
        artifacts["model_artifacts"].subgroup_uplift,
        artifacts["model_artifacts"].metrics,
        artifacts["model_artifacts"].calibration,
        artifacts["top_segments"],
    )
    summary = {
        "dataset_rows": int(len(artifacts["feature_df"])),
        "top_message": artifacts["experiment_summary"].iloc[0]["message_label"],
        "best_segment_message": artifacts["top_segments"].iloc[0]["best_message"],
        "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
