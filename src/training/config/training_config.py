from pathlib import Path
from dataclasses import dataclass

from src.preprocessing.config.preprocessing_config import (
    PreprocessingConfig
)

from src.preprocessing.config.augmentation_config import (
    AugmentationConfig
)


@dataclass
class TrainingConfig:

    experiment_name: str

    model_name: str

    preprocessing_config: PreprocessingConfig

    augmentation_config: AugmentationConfig

    batch_size: int

    epochs: int

    learning_rate: float

    learning_rate_display: str

    seed: int

    num_classes: int

    train_csv: Path

    val_csv: Path

    test_csv: Path

    trainable_backbone: bool = False