import numpy as np


def build_confidence_bins(confidences):

    if len(confidences) == 0:

        return {}

    hist, _ = np.histogram(
        confidences,
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    return {

        "0.0-0.5": int(hist[0]),

        "0.5-0.6": int(hist[1]),

        "0.6-0.7": int(hist[2]),

        "0.7-0.8": int(hist[3]),

        "0.8-0.9": int(hist[4]),

        "0.9-1.0": int(hist[5])
    }


def save_confidence_statistics(
    confidences
):

    if len(confidences) == 0:

        return None

    return {

        "mean":
            float(np.mean(confidences)),

        "std":
            float(np.std(confidences)),

        "min":
            float(np.min(confidences)),

        "max":
            float(np.max(confidences)),

        "median":
            float(np.median(confidences)),

        "q1":
            float(
                np.percentile(
                    confidences,
                    25
                )
            ),

        "q3":
            float(
                np.percentile(
                    confidences,
                    75
                )
            )
    }