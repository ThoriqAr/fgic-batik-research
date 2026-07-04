from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttributionSample:

    attribution_id: str

    attribution_path: Path

    image_path: Path

    ground_truth: str

    pred_label: str

    pred_index: int

    confidence: float