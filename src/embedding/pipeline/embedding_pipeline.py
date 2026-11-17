import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from datetime import datetime
from tensorflow.keras.models import load_model

from src.utils.path_config import (
    ARTIFACTS_DIR,
    PROCESSED_DIR
)

from src.training.artifacts.load_training_metadata import (
    load_training_metadata
)

from src.utils.preprocess_input import (
    get_preprocess_fn
)

from src.xai.loaders.load_single_image import (
    load_single_image
)

from src.embedding.generators.embedding_generator import (
    build_embedding_extractor,
    generate_embedding
)

from src.embedding.artifacts.embedding_artifacts import (
    save_embedding_database
)


def run_embedding(
    model_path,
    metadata_path,
    preprocessing_config,
    embedding_config
):

    print("\n[1/5] Load Training Metadata...")

    metadata = load_training_metadata(metadata_path)

    preprocess_fn = get_preprocess_fn(
        embedding_config.model_name
    )

    print(
        f"Loaded metadata : {metadata_path}"
    )


    print("\n[2/5] Load Model...")

    model = load_model(model_path)

    print(
        f"Loaded model : {model_path}"
    )


    print("\n[3/5] Build Embedding Extractor...")

    extractor = build_embedding_extractor(
        model=model,
        embedding_layer=embedding_config.embedding_layer
    )

    print(
        f"Embedding layer : {embedding_config.embedding_layer}"
    )


    print("\n[4/5] Generate Embeddings...")

    dataset_df = pd.read_csv(
        embedding_config.dataset_csv
    )

    embeddings = []

    labels = []

    image_paths = []


    dataset_pbar = tqdm(
        dataset_df.itertuples(index=False),
        total=len(dataset_df),
        desc="Generating Embeddings",
        unit="image"
    )


    for row in dataset_pbar:

        image_path = (
            PROCESSED_DIR /
            row.filepath
        )

        _, processed_image = (
            load_single_image(
                image_path=image_path,
                preprocessing_config=preprocessing_config,
                preprocess_fn=preprocess_fn
            )
        )

        embedding = generate_embedding(
            extractor,
            processed_image
        )[0]

        embeddings.append(embedding)

        labels.append(
            row.label
        )

        image_paths.append(
            str(image_path)
        )


    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    labels = np.asarray(labels)

    image_paths = np.asarray(
        image_paths
    )


    if embedding_config.normalize:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = (
            embeddings /
            np.clip(norms, 1e-8, None)
        )


    print("\n[5/5] Save Embedding Database...")


    experiment_id = (
        f"{embedding_config.experiment_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )


    output_dir = (
        ARTIFACTS_DIR /
        "embeddings" /
        experiment_id
    )


    metadata = {

        "model_name": embedding_config.model_name,

        "embedding_layer": embedding_config.embedding_layer,

        "embedding_dimension": int(embeddings.shape[1]),

        "normalize": embedding_config.normalize,

        "num_images": len(dataset_df),

        "model_path": model_path.as_posix(),

        "metadata_path": metadata_path.as_posix(),

        "dataset_csv": embedding_config.dataset_csv.as_posix(),

        "image_size": metadata["image_size"],

        "num_classes": metadata["num_classes"],

        "seed": metadata["seed"],

        "backbone_name": metadata["backbone_name"]

    }


    save_embedding_database(
        output_dir=output_dir,
        embeddings=embeddings,
        labels=labels,
        image_paths=image_paths,
        metadata=metadata
    )


    print()

    print("=" * 45)

    print("Embedding database generated")

    print("=" * 45)

    print(f"Output      : {output_dir}")

    print(f"Images      : {len(image_paths)}")

    print(f"Dimension   : {embeddings.shape[1]}")

    print(f"Normalized  : {embedding_config.normalize}")

    print("=" * 45)