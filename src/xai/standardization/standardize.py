import cv2
import numpy as np


def standardize_heatmap(
    heatmap,
    target_size
):

    heatmap = cv2.resize(
        heatmap,
        target_size
    )

    heatmap = heatmap.astype(
        np.float32
    )

    heatmap -= heatmap.min()

    heatmap /= (
        heatmap.max() + 1e-8
    )

    return heatmap