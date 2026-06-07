from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingConfig:

    experiment_name: str

    model_name: str

    image_size: tuple

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