import cv2
import numpy as np


def build_gradcam_heatmap(
    image,
    heatmap,
    threshold=None,
    gamma=1.0
):
    
    heatmap = cv2.resize(
        heatmap,
        (
            image.shape[1],
            image.shape[0]
        )
    )

    if threshold is not None:

        mask = (
            heatmap > threshold
        )

        heatmap_vis = np.zeros_like(
            heatmap
        )

        heatmap_vis[mask] = (
            heatmap[mask]
        )

    else:

        heatmap_vis = heatmap.copy()

    heatmap_vis = np.power(
        heatmap_vis,
        gamma
    )

    return heatmap_vis