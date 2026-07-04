from src.xai.loaders.load_single_image import (
    load_single_image
)


def load_evaluation_original_image(
    sample,
    model_context
):

    original_image, _ = (
        load_single_image(
            image_path=sample.image_path,
            preprocessing_config=model_context.preprocessing_config,
            preprocess_fn=model_context.preprocess_fn
        )
    )

    return original_image