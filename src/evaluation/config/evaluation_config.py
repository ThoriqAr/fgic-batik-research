from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvaluationConfig:

    experiment_name: str

    attribution_dir: Path

    metrics: list[str]

    perturbations: list[str]

    batch_size: int = 32

    seed: int = 42

    max_samples: int | None = None

    samples_per_class: int | None = None

    selected_classes: list[str] | None = None

    selected_attribution_ids: list[str] | None = None