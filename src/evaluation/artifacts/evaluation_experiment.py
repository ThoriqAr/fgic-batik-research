from pathlib import Path
from dataclasses import dataclass

from src.evaluation.artifacts.manifest import (
    EvaluationManifest
)


@dataclass
class EvaluationExperiment:

    experiment_name: str

    model_name: str

    xai_method: str

    attribution_dir: Path

    manifest: EvaluationManifest

    config: dict