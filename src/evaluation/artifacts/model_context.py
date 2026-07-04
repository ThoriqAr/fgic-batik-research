from dataclasses import dataclass

from src.preprocessing.config.preprocessing_config import (
    PreprocessingConfig
)


@dataclass
class ModelContext:

    model: object

    metadata: dict

    preprocess_fn: callable

    preprocessing_config: (
        PreprocessingConfig
    )