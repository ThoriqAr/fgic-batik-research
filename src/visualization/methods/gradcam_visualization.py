from src.visualization.loaders.image_loader import (
    load_visualization_image
)

from src.visualization.builders.gradcam.heatmap_builder import (
    build_gradcam_heatmap
)

from src.visualization.builders.gradcam.overlay_builder import (
    build_gradcam_overlay
)

from src.visualization.builders.gradcam.figure_builder import (
    build_gradcam_figure
)


def build_gradcam_visualization(
    attribution,
    alpha,
    gamma,
    threshold,
    contour_thickness,
    contour_levels_to_draw
):

    image = load_visualization_image(
        image_path=attribution["img_path"],
        image_size=attribution["image_size"],
        color_mode=attribution["color_mode"]
    )

    heatmap = build_gradcam_heatmap(
        image=image,
        heatmap=attribution["raw_heatmap"],
        threshold=threshold,
        gamma=gamma
    )

    overlay = build_gradcam_overlay(
        image=image,
        heatmap=heatmap,
        alpha=alpha,
        contour_thickness=contour_thickness,
        contour_levels_to_draw=contour_levels_to_draw
    )

    return build_gradcam_figure(
        image=image,
        heatmap=heatmap,
        overlay=overlay,
        attribution=attribution
    )