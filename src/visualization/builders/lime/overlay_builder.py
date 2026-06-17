import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_signed_heatmap(heatmap):

    max_abs = np.max(
        np.abs(heatmap)
    )

    if max_abs < 1e-8:

        return heatmap

    return heatmap / max_abs


def build_signed_overlay(
    image,
    heatmap,
    alpha=0.5
):

    heatmap = normalize_signed_heatmap(
        heatmap
    )

    cmap = plt.get_cmap(
        "seismic"
    )

    colored_heatmap = cmap(
        (heatmap + 1) / 2
    )

    colored_heatmap = (
        colored_heatmap[:, :, :3]
        * 255
    ).astype(np.uint8)

    mask = np.abs(
        heatmap
    ) > 1e-8

    overlay = image.copy()

    overlay[mask] = (

        (1 - alpha)
        * image[mask]

        +

        alpha
        * colored_heatmap[mask]

    ).astype(np.uint8)

    return overlay


def build_magnitude_overlay(
    image,
    heatmap,
    alpha=0.5
):

    if heatmap.max() > 0:

        heatmap = (
            heatmap /
            heatmap.max()
        )

    heatmap_uint8 = (
        heatmap * 255
    ).astype(np.uint8)

    colored_heatmap = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_INFERNO
        )
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        1 - alpha,
        colored_heatmap,
        alpha,
        0
    )

    return overlay


def build_lime_overlay(
    image,
    heatmap,
    mode,
    alpha=0.5
):

    if mode == "signed":

        return build_signed_overlay(
            image=image,
            heatmap=heatmap,
            alpha=alpha
        )

    return build_magnitude_overlay(
        image=image,
        heatmap=heatmap,
        alpha=alpha
    )