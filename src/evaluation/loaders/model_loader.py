from tensorflow.keras.models import (
    load_model
)

from src.training.artifacts.load_training_metadata import (
    load_training_metadata
)

from src.utils.preprocess_input import (
    get_preprocess_fn
)

from src.preprocessing.config.preprocessing_config import (
    PreprocessingConfig
)

from src.evaluation.artifacts.model_context import (
    ModelContext
)


def load_model_context(
    experiment
):

    metadata_path = (
        experiment.config["metadata_path"]
    )

    model_path = (
        experiment.config["model_path"]
    )

    metadata = (
        load_training_metadata(
            metadata_path
        )
    )

    model = load_model(
        model_path
    )

    preprocess_fn = (
        get_preprocess_fn(
            metadata["model_name"]
        )
    )

    preprocessing_config = (
        PreprocessingConfig(
            image_size=tuple(metadata["image_size"]),
            color_mode=metadata["color_mode"]
        )
    )

    return ModelContext(
        model=model,
        metadata=metadata,
        preprocess_fn=preprocess_fn,
        preprocessing_config=preprocessing_config
    )