import cv2
import numpy as np


def draw_contours(
    overlay,
    heatmap,
    contour_thickness,
    contour_levels_to_draw
):

    if not contour_levels_to_draw:

        return overlay

    contour_ranges = {

        "high": (0.70, 1.00),
        "medium": (0.45, 0.70),
        "low": (0.25, 0.45)
    }

    contour_colors = {

        "high": (0, 0, 255),
        "medium": (0, 165, 255),
        "low": (0, 255, 255)
    }

    for level_name in contour_levels_to_draw:

        lower, upper = (
            contour_ranges[level_name]
        )

        mask = np.logical_and(
            heatmap >= lower,
            heatmap < upper
        )

        mask = (
            mask.astype(np.uint8)
            * 255
        )

        contours, _ = (
            cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            contour_colors[level_name],
            contour_thickness
        )

    return overlay