import tensorflow as tf


class RandomBrightness(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        brightness_range=(0.9, 1.1),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.min_factor = brightness_range[0]
        self.max_factor = brightness_range[1]

    def call(
        self,
        images,
        training=None
    ):

        if not training:
            return images

        factor = tf.random.uniform(
            [1,1,1],
            self.min_factor,
            self.max_factor
        )

        images = images * factor

        images = tf.clip_by_value(
            images,
            0.0,
            255.0
        )

        return images