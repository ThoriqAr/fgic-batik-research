import tensorflow as tf


def preprocess_image(
    image,
    image_size=(224, 224),
    preprocessing_fn=None
):

    image = tf.image.resize(
        image,
        image_size
    )

    image = tf.cast(
        image,
        tf.float32
    )

    if preprocessing_fn is not None:
        image = preprocessing_fn(image)

    return image