from pathlib import Path

from src.visualization.builders.visualization_builder import (
    build_visualization
)

from src.visualization.artifacts.visualization import (
    Visualization
)

from src.visualization.loaders.manifest_loader import (
    filter_manifest
)


def visualize(
    manifest,
    classes=None,
    filenames=None,
    attribution_ids=None,
    max_samples=None,
    **kwargs
):

    selected = filter_manifest(
        manifest,
        classes=classes,
        filenames=filenames,
        attribution_ids=attribution_ids,
        max_samples=max_samples
    )

    for _, row in selected.iterrows():

        figure = build_visualization(
            npz_path=row["attribution_path"],
            **kwargs
        )

        if figure is None:
            continue

        yield Visualization(
            figure=figure,
            class_name=row["ground_truth"],
            npz_path=row["attribution_path"]
        )