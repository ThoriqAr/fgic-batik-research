import numpy as np

from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.xai.generators.attribution_generator import (
    generate_attribution
)

from src.evaluation.standardization.attribution_to_heatmap import (
    attribution_to_heatmap
)


def preprocess_perturbed_image(
    image,
    preprocess_fn
):

    image = image.astype(
        np.float32
    )

    image = preprocess_fn(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def load_original_attribution(
    sample
):

    return load_attribution(
        sample.attribution_path
    )


def load_perturbed_attribution(
    sample,
    perturbed_image,
    experiment,
    model_context
):
    
    attribution = (
        load_original_attribution(
            sample
        )
    )

    pred_index = int(
        attribution["pred_index"]
    )

    processed_image = (
        preprocess_perturbed_image(
            perturbed_image,
            model_context.preprocess_fn
        )
    )

    method_config = (
        attribution.get("method_config")
    )

    attribution_result = (
        generate_attribution(
            method=attribution["xai_method"],
            model=model_context.model,
            original_image=perturbed_image,
            processed_image=processed_image,
            pred_index=pred_index,
            backbone_name=attribution["backbone_name"],
            last_conv_layer_name=attribution["last_conv_layer_name"],
            preprocess_fn=model_context.preprocess_fn,
            target_class=pred_index,
            lime_config=method_config
        )
    )

    temp_attribution = {
        "xai_method" : attribution["xai_method"],
        "image_size" : attribution["image_size"]
    }

    temp_attribution.update(
        attribution_result
    )

    heatmap = (
        attribution_to_heatmap(
            temp_attribution
        )
    )

    return heatmap

