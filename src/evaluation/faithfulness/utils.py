import numpy as np

def preprocess_for_model(
    image,
    preprocess_fn
):

    image = image.astype(
        np.float32
    )

    image = preprocess_fn(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def predict_target_confidence(
    model,
    image,
    target_class,
    preprocess_fn
):

    image = (
        preprocess_for_model(
            image,
            preprocess_fn
        )
    )

    preds = model.predict(
        image,
        verbose=0
    )

    return float(
        preds[
            0,
            target_class
        ]
    )


def build_black_image(
    image
):

    return np.zeros_like(
        image,
        dtype=np.uint8
    )