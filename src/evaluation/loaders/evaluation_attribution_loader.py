import numpy as np

from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.evaluation.standardization.attribution_to_heatmap import (
    attribution_to_heatmap
)


def load_evaluation_attribution(
    sample
):

    attribution = (
        load_attribution(
            sample.attribution_path
        )
    )

    heatmap = (
        attribution_to_heatmap(
            attribution
        )
    )

    heatmap = np.expand_dims(
        heatmap,
        axis=0
    )

    return heatmap.astype(
        np.float32
    )