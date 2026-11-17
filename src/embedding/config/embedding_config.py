from pathlib import Path
from dataclasses import dataclass


@dataclass(slots=True)
class EmbeddingConfig:

    experiment_name: str

    model_name: str

    embedding_layer: str

    dataset_csv: Path

    batch_size: int = 32

    normalize: bool = True