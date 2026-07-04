from src.xai.explainers.gradcam import (
    generate_gradcam
)

from src.xai.explainers.lime import (
    generate_lime
)


def generate_attribution(
    method,
    model,
    original_image,
    processed_image,
    pred_index,
    backbone_name,
    last_conv_layer_name,
    preprocess_fn,
    target_class=None,
    lime_config=None
):

    method = method.lower()

    if method == "gradcam":

        return generate_gradcam(
            model=model,
            processed_image=processed_image,
            pred_index=pred_index,
            backbone_name=backbone_name,
            last_conv_layer_name=last_conv_layer_name
        )

    if method == "lime":

        target_class = (
            target_class
            if target_class is not None
            else pred_index
        )

        lime_config = (
            lime_config or {}
        )

        return generate_lime(
            model=model,
            original_image=original_image,
            preprocess_fn=preprocess_fn,
            target_class=target_class,
            **lime_config
        )

    raise ValueError(
        f"Unknown method: {method}"
    )