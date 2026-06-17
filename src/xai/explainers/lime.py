import numpy as np
from lime import lime_image
from skimage.segmentation import slic

def build_lime_explanation(
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

    return {
        "segments": segments,
        "local_exp": local_exp
    }


def generate_lime(
    model,
    original_image,
    preprocess_fn,
    num_samples=1000,
    n_segments=150,
    compactness=5,
    sigma=1,
    seed=42
):

    explanation = (
        build_lime_explanation(
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

    segments = explanation["segments"]

    local_exp = explanation["local_exp"]

    segment_ids = np.array(
        list(local_exp.keys()),
        dtype=np.int32
    )

    weights = np.array(
        list(local_exp.values()),
        dtype=np.float32
    )

    if len(weights) > 0:

        max_abs_weight = np.max(
            np.abs(weights)
        ).astype(np.float32)

    else:

        max_abs_weight = np.float32(0.0)

    return {

        "segments":
            segments.astype(
                np.int32
            ),

        "segment_ids":
            segment_ids,

        "weights":
            weights,

        "max_abs_weight":
            max_abs_weight

    }