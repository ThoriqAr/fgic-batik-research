import pandas as pd
import tensorflow as tf
from src.data.preparation.preprocess_image import preprocess_image

from src.utils.path_config import PROCESSED_DIR


def _load_sample(
    image_path,
    label,
    image_size,
    preprocessing_fn
):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = preprocess_image(
        image=image,
        image_size=image_size,
        preprocessing_fn=preprocessing_fn
    )

    return image, label


def load_dataset(
    csv_path,
    image_size=(224,224),
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
            image_size,
            preprocessing_fn
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.cache()

    ds = ds.batch(
        batch_size,
        drop_remainder=False
    )

    ds = ds.prefetch(
        tf.data.AUTOTUNE
    )

    return ds