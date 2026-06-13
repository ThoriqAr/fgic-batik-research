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
    target_size,
    lime_config=None
):

    method = method.lower()

    if method == "gradcam":

        return generate_gradcam(
            model=model,
            processed_image=processed_image,
            pred_index=pred_index,
            backbone_name=backbone_name,
            last_conv_layer_name=last_conv_layer_name,
            target_size=target_size
        )

    if method == "lime":

        lime_config = (
            lime_config or {}
        )

        return generate_lime(
            model=model,
            original_image=original_image,
            preprocess_fn=preprocess_fn,
            target_size=target_size,
            **lime_config
        )

    raise ValueError(
        f"Unknown method: {method}"
    )