import json
import keras
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils.calculate import (
    calculate_dataset_hash,
    calculate_file_hash,
    format_learning_rate
)

from src.utils.path_config import METADATA_DIR

class_mapping_path = (
    METADATA_DIR /
    "class_mapping.json"
)

def save_best_epoch(
    history,
    output_path
):

    best_epoch_idx = (
        history.history["val_loss"]
        .index(
            min(
                history.history["val_loss"]
            )
        )
    )

    summary = {

        "selected_metric": "val_loss",

        "selection_rule": "minimum_validation_loss",

        "best_epoch":
            best_epoch_idx + 1,

        "learning_rate": format_learning_rate(
            history.history["learning_rate"][best_epoch_idx]
        ),

        "train_accuracy":
            float(
                history.history["accuracy"][best_epoch_idx]
            ),

        "train_loss":
            float(
                history.history["loss"][best_epoch_idx]
            ),

        "val_accuracy":
            float(
                history.history["val_accuracy"][best_epoch_idx]
            ),

        "val_loss":
            float(
                history.history["val_loss"][best_epoch_idx]
            ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


def save_metadata(
    output_path,
    train_config
):

    metadata = {

        "experiment_name":
            train_config.experiment_name,

        "model_name":
            train_config.model_name,

        "image_size":
            list(train_config.preprocessing_config.image_size),

        "color_mode":
            train_config.preprocessing_config.color_mode,

        "augmentation":{

            "preset_name":
                train_config
                .augmentation_config
                .preset_name,

            "rotation":
                train_config
                .augmentation_config
                .rotation,

            "width_shift":
                train_config
                .augmentation_config
                .width_shift,

            "height_shift":
                train_config
                .augmentation_config
                .height_shift,

            "zoom":
                train_config
                .augmentation_config
                .zoom,

            "brightness":
                train_config
                .augmentation_config
                .brightness
        },

        "batch_size":
            train_config.batch_size,

        "epochs":
            train_config.epochs,

        "learning_rate":
            train_config.learning_rate,

        "learning_rate_display":
            train_config.learning_rate_display,

        "seed":
            train_config.seed,

        "num_classes":
            train_config.num_classes,

        "trainable_backbone":
            train_config.trainable_backbone,

        "backbone_total_layers":
            train_config.backbone_total_layers,

        "unfreeze_last_n_layers":
            train_config.unfreeze_last_n_layers,

        "trainable_backbone_layers":
            train_config.trainable_backbone_layers,

        "fine_tune_stage":
            train_config.fine_tune_stage,

        "fine_tune_from":
            (
                train_config
                .fine_tune_from
                .as_posix()
                if train_config.fine_tune_from
                else None
            ),

        "train_csv":
            train_config.train_csv.as_posix(),

        "val_csv":
            train_config.val_csv.as_posix(),

        "test_csv":
            train_config.test_csv.as_posix(),

        "train_csv_hash":
            calculate_file_hash(
                train_config.train_csv
            ),

        "val_csv_hash":
            calculate_file_hash(
                train_config.val_csv
            ),

        "test_csv_hash":
            calculate_file_hash(
                train_config.test_csv
            ),

        "dataset_hash":
            calculate_dataset_hash(
                train_config.train_csv,
                train_config.val_csv,
                train_config.test_csv
            ),

        "class_mapping_file":
            class_mapping_path.as_posix(),

        "class_mapping_hash":
            calculate_file_hash(
                class_mapping_path
            ),

        "created_at":
            datetime.now().isoformat(),

        "tensorflow_version":
            tf.__version__,

        "keras_version":
            keras.__version__,

    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


def save_learning_curves(
    history_csv_path,
    output_path
):
    
    df = pd.read_csv(
        history_csv_path
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        df["accuracy"]
    )

    axes[0].plot(
        df["val_accuracy"]
    )

    axes[0].set_title(
        "Accuracy"
    )

    axes[0].legend(
        ["train", "validation"]
    )

    axes[1].plot(
        df["loss"]
    )

    axes[1].plot(
        df["val_loss"]
    )

    axes[1].set_title(
        "Loss"
    )

    axes[1].legend(
        ["train", "validation"]
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


def format_history_csv(history_path, output_path):

    df = pd.read_csv(
        history_path
    )

    df["epoch"] = (
        df["epoch"] + 1
    )

    df["learning_rate"] = (
        df["learning_rate"]
        .apply(format_learning_rate)
    )

    columns = [

        "epoch",

        "learning_rate",

        "accuracy",

        "loss",

        "val_accuracy",

        "val_loss"
    ]

    df = df[
        columns
    ]

    df.to_csv(
        output_path,
        index=False
    )