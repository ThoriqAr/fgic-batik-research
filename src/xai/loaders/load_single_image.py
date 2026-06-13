import tensorflow as tf

from src.data.loaders.load_image import (
    load_image
)


def load_single_image(
    image_path,
    preprocessing_config,
    preprocess_fn
):
    
    image = load_image(
        image_path,
        preprocessing_config
    )

    visualization_image = (
        image.numpy()
        .astype("uint8")
    )

    processed_image = (
        preprocess_fn(
            image
        )
        .numpy()
    )

    processed_image = (
        processed_image[None, ...]
    )

    return (
        visualization_image,
        processed_image
    )