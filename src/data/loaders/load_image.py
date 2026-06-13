from pathlib import Path
import tensorflow as tf

from src.data.preparation.preprocess_image import (
    preprocess_image
)


def load_image(
    image_path,
    preprocessing_config
):

    image_path = str(
        Path(image_path)
    )

    image = tf.io.read_file(
        image_path
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = preprocess_image(
        image=image,
        preprocessing_config=preprocessing_config
    )

    return image