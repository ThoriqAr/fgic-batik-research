from pathlib import Path
from dataclasses import dataclass

from src.preprocessing.config.preprocessing_config import (
    PreprocessingConfig
)

@dataclass
class EvaluationConfig:

    model_name: str

    preprocessing_config: PreprocessingConfig

    batch_size: int

    seed: int

    num_classes: int

    test_csv: Path