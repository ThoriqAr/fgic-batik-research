import json
import numpy as np


def save_embedding_database(
    output_dir,
    embeddings,
    labels,
    image_paths,
    metadata
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_dir / "embeddings.npy",
        embeddings
    )

    np.save(
        output_dir / "labels.npy",
        labels
    )

    np.save(
        output_dir / "image_paths.npy",
        image_paths
    )

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )