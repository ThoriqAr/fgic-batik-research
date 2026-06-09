import pandas as pd
import tensorflow as tf

from src.utils.path_config import PROCESSED_DIR

from src.data.preparation.preprocess_image import (
    preprocess_image
)


def _load_sample(
    image_path,
    label,
    preprocessing_config
):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = preprocess_image(
        image=image,
        preprocessing_config=preprocessing_config
    )

    return image, label


def load_dataset(
    csv_path,
    preprocessing_config,
    augmentation_layer=None,
    is_training=False,
    batch_size=32,
    shuffle=False,
    seed=42,
    num_classes=20,
    preprocessing_fn=None
):

    df = pd.read_csv(csv_path)

    image_paths = [
        str(PROCESSED_DIR / p)
        for p in df["filepath"]
    ]

    labels = tf.keras.utils.to_categorical(
        df["class_id"],
        num_classes=num_classes
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    options = tf.data.Options()

    options.experimental_deterministic = True

    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=seed
        )

    ds = ds.map(
        lambda x, y: _load_sample(
            x,
            y,
            preprocessing_config,
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.cache()

    if (
        is_training
        and
        augmentation_layer is not None
    ):

        ds = ds.map(
            lambda x, y: (
                augmentation_layer(
                    x,
                    training=True
                ),
                y
            ),
            num_parallel_calls=
                tf.data.AUTOTUNE
        )

    if preprocessing_fn is not None:

        ds = ds.map(
            lambda x, y: (
                preprocessing_fn(x),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(
        batch_size,
        drop_remainder=False
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds