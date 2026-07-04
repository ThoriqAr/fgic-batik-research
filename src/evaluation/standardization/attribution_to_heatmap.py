import cv2
import numpy as np

from src.evaluation.standardization.normalize import (
    minmax_normalize
)


def build_gradcam_heatmap(
    attribution,
    image_size
):

    heatmap = attribution[
        "raw_heatmap"
    ]

    heatmap = cv2.resize(

        heatmap,

        (
            image_size[1],
            image_size[0]
        )
    )

    return heatmap.astype(
        np.float32
    )


def build_lime_heatmap(
    attribution
):

    segments = attribution[
        "segments"
    ]

    segment_ids = attribution[
        "segment_ids"
    ]

    weights = attribution[
        "weights"
    ]

    heatmap = np.zeros(
        segments.shape,
        dtype=np.float32
    )

    for segment_id, weight in zip(

        segment_ids,
        weights

    ):

        heatmap[
            segments == segment_id
        ] = weight

    return heatmap


def attribution_to_heatmap(
    attribution,
    normalize=True
):

    method = attribution[
        "xai_method"
    ].lower()

    image_size = tuple(
        attribution[
            "image_size"
        ]
    )

    if method == "gradcam":

        heatmap = (
            build_gradcam_heatmap(
                attribution,
                image_size
            )
        )

    elif method == "lime":

        heatmap = (
            build_lime_heatmap(
                attribution
            )
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    if normalize:

        heatmap = (
            minmax_normalize(
                heatmap
            )
        )

    return heatmap