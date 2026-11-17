import matplotlib.pyplot as plt

from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.visualization.loaders.image_loader import (
    load_visualization_image
)

from src.visualization.builders.gradcam.heatmap_builder import (
    build_gradcam_heatmap
)

from src.visualization.builders.gradcam.overlay_builder import (
    build_gradcam_overlay
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


def build_overlay_figure(
    attribution_path,
    gradcam_kwargs,
    lime_kwargs
):

    attribution = load_attribution(
        attribution_path
    )

    image = load_visualization_image(
        image_path=attribution["img_path"],
        image_size=attribution["image_size"],
        color_mode=attribution["color_mode"]
    )

    method = attribution[
        "xai_method"
    ].lower()

    if method == "gradcam":

        heatmap = build_gradcam_heatmap(
            image=image,
            heatmap=attribution["raw_heatmap"],
            threshold=gradcam_kwargs["threshold"],
            gamma=gradcam_kwargs["gamma"]
        )

        overlay = build_gradcam_overlay(
            image=image,
            heatmap=heatmap,
            alpha=gradcam_kwargs["alpha"],
            contour_thickness=gradcam_kwargs[
                "contour_thickness"
            ],
            contour_levels_to_draw=gradcam_kwargs[
                "contour_levels_to_draw"
            ]
        )

    elif method == "lime":

        selected_segments = (
            select_segments(
                segment_ids=attribution[
                    "segment_ids"
                ],
                weights=attribution[
                    "weights"
                ],
                mode=lime_kwargs["mode"],
                top_k=lime_kwargs["top_k"]
            )
        )

        heatmap = build_lime_heatmap(
            segments=attribution[
                "segments"
            ],
            selected_segments=selected_segments,
            mode=lime_kwargs["mode"]
        )

        overlay = build_lime_overlay(
            image=image,
            heatmap=heatmap,
            mode=lime_kwargs["mode"],
            alpha=lime_kwargs["alpha"]
        )

    else:

        raise ValueError(
            f"Unknown method : {method}"
        )

    fig = plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        overlay
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    return fig