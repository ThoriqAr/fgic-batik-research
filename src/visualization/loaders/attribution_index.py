from pathlib import Path
from collections import defaultdict


def build_attribution_index(
    attribution_dir
):

    attribution_dir = Path(
        attribution_dir
    )

    grouped = defaultdict(list)

    for npz_path in sorted(
        attribution_dir.glob("*.npz")
    ):

        class_name = (
            npz_path.stem
            .rsplit("_", 1)[0]
        )

        grouped[
            class_name
        ].append(npz_path)

    return dict(grouped)