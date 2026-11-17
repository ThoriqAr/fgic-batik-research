from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.qualitative.builders.build_original_figure import (
    build_original_figure
)

from src.qualitative.builders.build_overlay_figure import (
    build_overlay_figure
)

from src.qualitative.models.qualitative_visualization import (
    QualitativeVisualization
)


def build_qualitative_figure(
    sample,
    gradcam_kwargs,
    lime_kwargs
):

    original_attribution = (
        load_attribution(
            sample.mobilenet_gradcam
        )
    )

    original = (
        build_original_figure(
            attribution=original_attribution
        )
    )

    explanation_a = (
        build_overlay_figure(
            attribution_path=sample.mobilenet_gradcam,
            gradcam_kwargs=gradcam_kwargs,
            lime_kwargs=lime_kwargs
        )
    )

    explanation_b = (
        build_overlay_figure(
            attribution_path=sample.mobilenet_lime,
            gradcam_kwargs=gradcam_kwargs,
            lime_kwargs=lime_kwargs
        )
    )

    explanation_c = (
        build_overlay_figure(
            attribution_path=sample.resnet_gradcam,
            gradcam_kwargs=gradcam_kwargs,
            lime_kwargs=lime_kwargs
        )
    )

    explanation_d = (
        build_overlay_figure(
            attribution_path=sample.resnet_lime,
            gradcam_kwargs=gradcam_kwargs,
            lime_kwargs=lime_kwargs
        )
    )

    return QualitativeVisualization(

        sample=sample,

        original=original,

        explanation_a=explanation_a,

        explanation_b=explanation_b,

        explanation_c=explanation_c,

        explanation_d=explanation_d

    )