from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from src.utils.path_config import (
    ARTIFACTS_DIR
)


@dataclass
class EvaluationArtifacts:

    experiment_dir: Path

    results_csv: Path

    summary_json: Path

    config_json: Path

    errors_csv: Path


def create_evaluation_artifacts(
    experiment_name
):
  
    evaluation_id = (
        f"{experiment_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    experiment_dir = (
        ARTIFACTS_DIR /
        "evaluations" /
        evaluation_id
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return EvaluationArtifacts(

        experiment_dir=
            experiment_dir,

        results_csv=
            experiment_dir /
            "results.csv",

        summary_json=
            experiment_dir /
            "summary.json",

        config_json=
            experiment_dir /
            "config.json",

        errors_csv=
            experiment_dir /
            "errors.csv"
    )