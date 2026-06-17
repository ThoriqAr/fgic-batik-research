import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from tensorflow.keras.models import load_model

from src.utils.path_config import (
    PROCESSED_DIR,
    ARTIFACTS_DIR
)

from src.data.loaders.load_class_mapping import (
    get_class_names
)

from src.utils.preprocess_input import (
    get_preprocess_fn
)

from src.xai.prediction.predictor import (
    predict_image
)

from src.training.artifacts.load_training_metadata import (
    load_training_metadata
)

from src.xai.loaders.load_single_image import (
    load_single_image
)

from src.xai.generators.attribution_generator import (
    generate_attribution
)

from src.xai.artifacts.attribution import (
    create_attribution_id,
    save_attribution
)

from src.xai.artifacts.manifest import (
    create_manifest_record,
    save_manifest
)

from src.xai.artifacts.statistics import (
    save_confidence_statistics,
    build_confidence_bins
)

from src.xai.artifacts.experiment import (
    build_experiment_config,
    save_experiment_config
)


def run_attribution(
    model_path,
    metadata_path,
    evaluation_config,
    attribution_config,
    lime_config=None
):
  
    print("\n[1/5] Load Training Metadata...")

    metadata = load_training_metadata(
        metadata_path
    )

    print(f"Loaded Training Metadata: {metadata_path}")

    class_names = get_class_names()

    preprocess_fn = get_preprocess_fn(
        evaluation_config.model_name
    )

    print("\n[2/5] Load Model...")

    model = load_model(
        model_path
    )

    print(
        f"Loaded Model\n"
        f"Model Name    : {evaluation_config.model_name}\n"
        f"Model Path    : {model_path}"
    )

    print("\n[3/5] Load Test Set...")

    test_df = pd.read_csv(
        evaluation_config.test_csv
    )

    print(
        f"Loaded Test Set\n"
        f"Total Images  : {len(test_df)} Images\n"
        f"Test CSV Path : {evaluation_config.test_csv}"
    )

    print(
        f"\n[4/5] Generate Attribution Maps"
    )

    experiment_id = (
        f"{attribution_config.experiment_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = (
        ARTIFACTS_DIR /
        "attributions" /
        experiment_id
    )

    npz_dir = (
        output_dir /
        "attribution_maps"
    )

    npz_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    confidences = []

    filtered_images = 0

    error_images = 0

    class_distribution_saved = {

        class_name: 0
        for class_name in class_names
    }

    class_distribution_filtered = {

        class_name: 0
        for class_name in class_names
    }

    method_config = None

    if attribution_config.xai_method == "lime":
        method_config = (
            vars(lime_config)
            if lime_config
            else None
        )

    dataset_pbar = tqdm(
        test_df.itertuples(index=False),
        total=len(test_df),
        desc="Generated Attribution Maps",
        unit="image",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    for row in dataset_pbar:

        try:
        
            ground_truth = class_names[
                row.class_id
            ]

            image_path = (
                PROCESSED_DIR /
                row.filepath
            )

            visualization_image, processed_image = (
                load_single_image(
                    image_path=image_path,
                    preprocessing_config=evaluation_config.preprocessing_config,
                    preprocess_fn=preprocess_fn
                )
            )

            pred_index, confidence = (
                predict_image(
                    model,
                    processed_image
                )
            )

            pred_label = class_names[
                pred_index
            ]

            keep = True

            if attribution_config.correct_prediction_only:

                keep &= (
                    pred_label == ground_truth
                )

            if attribution_config.min_confidence is not None:

                keep &= (
                    confidence >=
                    attribution_config.min_confidence
                )

            if not keep:

                filtered_images += 1

                class_distribution_filtered[
                    ground_truth
                ] += 1

                continue

            attribution_result = (
                generate_attribution(
                    method=attribution_config.xai_method,
                    model=model,
                    original_image=visualization_image,
                    processed_image=processed_image,
                    pred_index=pred_index,
                    backbone_name=metadata["backbone_name"],
                    last_conv_layer_name=metadata["last_conv_layer_name"],
                    preprocess_fn=preprocess_fn,
                    lime_config=
                        vars(lime_config)
                        if lime_config
                        else None
                )
            )

            attribution_id = (
                create_attribution_id(
                    ground_truth,
                    str(image_path)
                )
            )

            save_path = (
                npz_dir /
                f"{attribution_id}.npz"
            )

            save_attribution(
                output_path=save_path,
                metadata=metadata,
                attribution_config=attribution_config,
                preprocess_fn=preprocess_fn,
                attribution_id=attribution_id,
                attribution_result=attribution_result,
                pred_index=pred_index,
                pred_label=pred_label,
                confidence=confidence,
                ground_truth=ground_truth,
                image_path=image_path,
                method_config=method_config
            )

            record = create_manifest_record(
                attribution_id=attribution_id,
                image_path=image_path,
                ground_truth=ground_truth,
                pred_label=pred_label,
                pred_index=pred_index,
                confidence=confidence,
                attribution_path=save_path
            )

            records.append(record)

            confidences.append(
                confidence
            )

            class_distribution_saved[
                ground_truth
            ] += 1

        except Exception as e:

            error_images += 1

            print(
                f"ERROR [{error_images}] "
                f"{row.filepath}"
            )

            print(repr(e))

            continue


    print("\n[5/5] Save artifacts...")

    if confidences:

        confidence_stats = (
            save_confidence_statistics(
                confidences
            )
        )

        confidence_bins = (
            build_confidence_bins(
                confidences
            )
        )

    else:

        confidence_stats = None
        confidence_bins = None

    save_manifest(
        records,
        output_dir /
        "manifest.csv"
    )

    experiment_config = build_experiment_config(
        metadata=metadata,
        attribution_config=attribution_config,
        model_path=model_path,
        metadata_path=metadata_path,
        preprocess_fn=preprocess_fn,
        class_names=class_names,
        total_images=len(test_df),
        saved_images=len(records),
        filtered_images=filtered_images,
        error_images=error_images,
        confidence_stats=confidence_stats,
        confidence_bins=confidence_bins,
        class_distribution_saved=class_distribution_saved,
        class_distribution_filtered=class_distribution_filtered,
        method_config=method_config
    )

    save_experiment_config(
        experiment_config,
        output_dir / "config.json"
    )

    print(
        f"\nGenerate attribution completed : {output_dir}"
    )

    saved_rate = (
        len(records) /
        len(test_df)
    ) * 100

    print("\n========== Attribution Summary ==========")

    print(f"Experiment ID   : {experiment_id}")
    print(f"Total Images    : {len(test_df)}")
    print(
        f"Saved Images    : "
        f"{len(records)} ({saved_rate:.2f}%)"
    )
    print(f"Filtered Images : {filtered_images}")
    print(f"Error Images    : {error_images}")

    print("========================================")