from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.evaluation.standardization.attribution_to_heatmap import (
    attribution_to_heatmap
)


def load_evaluation_heatmap(
    sample
):

    attribution = (
        load_attribution(
            sample.attribution_path
        )
    )

    return (
        attribution_to_heatmap(
            attribution
        )
    )