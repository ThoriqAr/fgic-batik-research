import numpy as np


def build_pixel_ranking(
    heatmap
):

    importance = np.abs(
        heatmap
    )

    ranking = np.argsort(
        importance.flatten()
    )[::-1]

    return ranking.astype(
        np.int64
    )