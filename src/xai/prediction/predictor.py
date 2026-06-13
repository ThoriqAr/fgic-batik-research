import numpy as np


def predict_image(
    model,
    image_batch
):

    preds = model.predict(
        image_batch,
        verbose=0
    )

    pred_index = int(
        np.argmax(preds[0])
    )

    confidence = float(
        preds[0][pred_index]
    )

    return (
        pred_index,
        confidence
    )