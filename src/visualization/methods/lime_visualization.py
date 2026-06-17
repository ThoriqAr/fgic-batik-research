from src.visualization.loaders.image_loader import (
    load_visualization_image
)

from src.visualization.builders.lime.segments_selectors import (
    select_segments
)

from src.visualization.builders.lime.heatmap_builder import (
    build_lime_heatmap
)

from src.visualization.builders.lime.overlay_builder import (
    build_lime_overlay
)

from src.visualization.builders.lime.figure_builder import (
    build_lime_figure
)


def build_lime_visualization(
    attribution,
    mode="positive",
    top_k=None,
    alpha=0.5
):

    image = load_visualization_image(
        image_path=attribution["img_path"],
        image_size=attribution["image_size"],
        color_mode=attribution["color_mode"]
    )

    selected_segments = select_segments(
        segment_ids=attribution["segment_ids"],
        weights=attribution["weights"],
        mode=mode,
        top_k=top_k
    )

    heatmap = build_lime_heatmap(
        segments=attribution["segments"],
        selected_segments=selected_segments,
        mode=mode
    )

    overlay = build_lime_overlay(
        image=image,
        heatmap=heatmap,
        mode=mode,
        alpha=alpha
    )

    return build_lime_figure(
        image=image,
        heatmap=heatmap,
        overlay=overlay,
        attribution=attribution,
        mode=mode
    )