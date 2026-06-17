from pathlib import Path

from src.visualization.builders.visualization_builder import (
    build_visualization
)

from src.visualization.artifacts.visualization import (
    Visualization
)


def visualize(
    attributions,
    classes=None,
    max_samples=None,
    **kwargs
):

    if classes is None:

        selected = attributions

    else:

        selected = {

            cls: attributions[cls]

            for cls in classes

            if cls in attributions

        }

    for class_name, npz_paths in selected.items():

        npz_paths = sorted(
            map(Path, npz_paths)
        )

        if max_samples is not None:

            npz_paths = npz_paths[:max_samples]

        for npz_path in npz_paths:

            figure = build_visualization(
                npz_path=npz_path,
                **kwargs
            )

            if figure is None:
                continue

            yield Visualization(
                figure=figure,
                class_name=class_name,
                npz_path=npz_path
            )