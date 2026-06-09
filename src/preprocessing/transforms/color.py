import tensorflow as tf


def apply_color_mode(
    image,
    color_mode
):

    if color_mode == "rgb":

        return image

    if color_mode == "grayscale":

        image = tf.image.rgb_to_grayscale(
            image
        )

        image = tf.image.grayscale_to_rgb(
            image
        )

        return image

    raise ValueError(
        f"Unknown color mode: {color_mode}"
    )