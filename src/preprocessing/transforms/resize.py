import tensorflow as tf


def resize_image(
    image,
    image_size
):

    return tf.image.resize(
        image,
        image_size
    )