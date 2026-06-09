import tensorflow as tf

from src.preprocessing.pipeline.preprocessing_pipeline import (
    apply_preprocessing
)


def preprocess_image(
    image,
    preprocessing_config
):

    image = apply_preprocessing(
        image,
        preprocessing_config
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image