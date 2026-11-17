import json
from datetime import datetime


def build_experiment_config(
    metadata,
    attribution_config,
    model_path,
    metadata_path,
    preprocess_fn,
    class_names,
    total_images,
    saved_images,
    filtered_images,
    error_images,
    runtime_seconds,
    confidence_stats,
    confidence_bins,
    class_distribution_saved,
    class_distribution_filtered,
    method_config=None
):

    runtime_minutes = (
        runtime_seconds / 60
    )

    samples_per_second = (
        saved_images / runtime_seconds
        if runtime_seconds > 0
        else 0.0
    )

    samples_per_minute = (
        saved_images / runtime_minutes
        if runtime_minutes > 0
        else 0.0
    )

    seconds_per_sample = (
        runtime_seconds / saved_images
        if saved_images > 0
        else 0.0
    )

    return {

        "experiment_name":
            attribution_config.experiment_name,

        "training_experiment_name":
            metadata["experiment_name"],

        "model_name":
            metadata["model_name"],

        "model_path":
            model_path.as_posix(),

        "metadata_path":
            metadata_path.as_posix(),

        "xai_method":
            attribution_config.xai_method,

        "image_size":
            metadata["image_size"],

        "color_mode":
            metadata["color_mode"],

        "seed":
            metadata["seed"],

        "backbone_name":
            metadata["backbone_name"],

        "last_conv_layer_name":
            metadata["last_conv_layer_name"],

        "preprocess_function":
            preprocess_fn.__name__,

        "filter_rule": {

            "correct_prediction_only":
                attribution_config.correct_prediction_only,

            "min_confidence":
                attribution_config.min_confidence
        },

        "class_names":
            class_names,

        "total_images":
            total_images,

        "saved_images":
            saved_images,

        "filtered_images":
            filtered_images,

        "error_images":
            error_images,

        "runtime_seconds":
            float(runtime_seconds),

        "runtime_minutes":
            float(runtime_minutes),

        "samples_per_second":
            float(samples_per_second),

        "samples_per_minute":
            float(samples_per_minute),

        "seconds_per_sample":
            float(seconds_per_sample),

        "confidence_statistics":
            confidence_stats,

        "confidence_bins":
            confidence_bins,

        "class_distribution_saved":
            class_distribution_saved,

        "class_distribution_filtered":
            class_distribution_filtered,

        "method_config":
            method_config,

        "dataset_hash":
            metadata["dataset_hash"],

        "class_mapping_hash":
            metadata["class_mapping_hash"],

        "created_at":
            datetime.now().isoformat()
    }


def save_experiment_config(
    config,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )