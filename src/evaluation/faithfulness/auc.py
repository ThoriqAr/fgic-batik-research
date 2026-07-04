from sklearn.metrics import auc


def compute_auc(
    x,
    y
):

    return float(
        auc(
            x,
            y
        )
    )