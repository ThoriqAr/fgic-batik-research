import tensorflow as tf

from src.preprocessing.transforms.brightness import (
    RandomBrightness
)


def build_augmentation_layer(
    augmentation_config
):

    layers = []

    if augmentation_config.rotation > 0:

        layers.append(
            tf.keras.layers.RandomRotation(
                augmentation_config.rotation / 360
            )
        )

    if (
        augmentation_config.width_shift > 0
        or
        augmentation_config.height_shift > 0
    ):

        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=
                    augmentation_config.height_shift,

                width_factor=
                    augmentation_config.width_shift
            )
        )

    if augmentation_config.zoom > 0:

        layers.append(
            tf.keras.layers.RandomZoom(
                augmentation_config.zoom
            )
        )

    if (
    augmentation_config.brightness
        is not None
    ):

        layers.append(
            RandomBrightness(
                augmentation_config.brightness
            )
        )

    if len(layers) == 0:
        
        return None

    return tf.keras.Sequential(
        layers,
        name="augmentation"
    )