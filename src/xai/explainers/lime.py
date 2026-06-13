import numpy as np
from lime import lime_image
from skimage.segmentation import slic

from src.xai.standardization.standardize import (
    standardize_heatmap
)


def build_lime_heatmap(
    model,
    original_image,
    preprocess_fn,
    num_samples=1000,
    n_segments=150,
    compactness=5,
    sigma=1,
    seed=42
):

    explainer = (
        lime_image.LimeImageExplainer(
            random_state=seed
        )
    )

    def predict_fn(images):

        images = np.asarray(
            images,
            dtype=np.float32
        )

        images = preprocess_fn(
            images
        )

        return model.predict(
            images,
            verbose=0
        )

    def segmentation_fn(image):

        return slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=0
        )
    
    explanation = (
        explainer.explain_instance(
            image=original_image.astype(
                np.double
            ),
            classifier_fn=predict_fn,
            top_labels=1,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn
        )
    )

    top_label = (
        explanation.top_labels[0]
    )

    segments = (
        explanation.segments
    )

    local_exp = dict(
        explanation.local_exp[
            top_label
        ]
    )

    heatmap = np.zeros_like(
        segments,
        dtype=np.float32
    )

    for seg_id, weight in local_exp.items():

        heatmap[
            segments == seg_id
        ] = abs(weight)

    heatmap /= (
        heatmap.max() + 1e-8
    )

    return heatmap


def generate_lime(
    model,
    original_image,
    preprocess_fn,
    target_size,
    num_samples=1000,
    n_segments=150,
    compactness=5,
    sigma=1,
    seed=42
):

    heatmap = (
        build_lime_heatmap(
            model=model,
            original_image=original_image,
            preprocess_fn=preprocess_fn,
            num_samples=num_samples,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            seed=seed
        )
    )

    return standardize_heatmap(
        heatmap,
        target_size
    )