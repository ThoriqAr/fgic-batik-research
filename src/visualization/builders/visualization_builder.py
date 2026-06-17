from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.visualization.methods.gradcam_visualization import (
    build_gradcam_visualization
)

from src.visualization.methods.lime_visualization import (
    build_lime_visualization
)


def build_visualization(
    npz_path,
    **kwargs
):

    attribution = load_attribution(npz_path)

    method = attribution[
        "xai_method"
    ].lower()

    if method == "gradcam":

        return build_gradcam_visualization(
            attribution=attribution,
            **kwargs
        )

    if method == "lime":

        return build_lime_visualization(
            attribution=attribution,
            **kwargs
        )

    raise ValueError(
        f"Unknown method: {method}"
    )