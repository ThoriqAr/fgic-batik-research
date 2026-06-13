import hashlib
import numpy as np
from pathlib import Path


def create_attribution_id(
    class_name,
    image_path
):

    file_hash = hashlib.md5(
        image_path.encode()
    ).hexdigest()[:16]

    return (
        f"{class_name}_{file_hash}"
    )


def save_attribution(
    output_path,
    metadata,
    attribution_config,
    preprocess_fn,
    attribution_id,
    heatmap,
    pred_index,
    pred_label,
    confidence,
    ground_truth,
    image_path,
    method_config=None
):

    attribution_data = {

        "attribution_id": attribution_id,

        "experiment_name":
            metadata["experiment_name"],

        "model_name":
            metadata["model_name"],

        "xai_method":
            attribution_config.xai_method,

        "backbone_name":
            metadata["backbone_name"],

        "last_conv_layer_name":
            metadata["last_conv_layer_name"],

        "method_config":
            method_config,

        "preprocess_function":
            preprocess_fn.__name__,

        "image_size":
            metadata["image_size"],

        "color_mode":
            metadata["color_mode"],

        "heatmap":
            heatmap.astype("float32"),

        "pred_index":
            pred_index,

        "pred_label":
            pred_label,

        "confidence":
            confidence,

        "ground_truth":
            ground_truth,

        "file_name":
            Path(image_path).name,

        "img_path":
            image_path.as_posix(),

        "dataset_hash":
            metadata["dataset_hash"],

        "class_mapping_hash":
            metadata["class_mapping_hash"]
    }

    np.savez_compressed(
        output_path,
        **attribution_data
    )