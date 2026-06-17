import numpy as np


def build_lime_heatmap(
    segments,
    selected_segments,
    mode
):

    heatmap = np.zeros_like(
        segments,
        dtype=np.float32
    )

    if len(selected_segments) == 0:
        
        return heatmap

    for seg_id, weight in selected_segments:

        value = (
            weight
            if mode == "signed"
            else abs(weight)
        )

        heatmap[
            segments == seg_id
        ] = value

    return heatmap