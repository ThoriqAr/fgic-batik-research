import numpy as np


def minmax_normalize(
    heatmap
):

    heatmap = heatmap.astype(
        np.float32
    )

    min_value = np.min(
        heatmap
    )

    max_value = np.max(
        heatmap
    )

    denominator = (
        max_value -
        min_value
    )

    if denominator <= 1e-8:

        return np.zeros_like(
            heatmap,
            dtype=np.float32
        )

    return (
        (heatmap - min_value)
        /
        denominator
    )