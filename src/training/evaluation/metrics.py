import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


def evaluate_model(
    model,
    test_ds,
    class_names
):

    y_prob = model.predict(
        test_ds,
        verbose=1
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    y_true = np.concatenate([
        np.argmax(y.numpy(), axis=1)
        for _, y in test_ds
    ])

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return report, cm