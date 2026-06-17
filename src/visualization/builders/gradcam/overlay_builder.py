import cv2

from src.visualization.builders.gradcam.contour import (
    draw_contours
)


def build_gradcam_overlay(
    image,
    heatmap,
    alpha=0.5,
    contour_thickness=1,
    contour_levels_to_draw=None
):

    heatmap_uint8 = (
        heatmap * 255
    ).astype("uint8")

    heatmap_color = (
        cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_INFERNO
        )
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    overlay = draw_contours(
        overlay=overlay,
        heatmap=heatmap,
        contour_thickness=contour_thickness,
        contour_levels_to_draw=contour_levels_to_draw
    )

    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay