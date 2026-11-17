from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class QualitativeSample:

    sample_id: str

    ground_truth: str

    file_name: str

    mobilenet_gradcam: Path

    mobilenet_lime: Path

    resnet_gradcam: Path

    resnet_lime: Path