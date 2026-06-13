from dataclasses import dataclass


@dataclass
class AttributionConfig:

    experiment_name: str

    xai_method: str

    correct_prediction_only: bool = True

    min_confidence: float | None = None