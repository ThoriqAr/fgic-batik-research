import json
import keras
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

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
        history.history["val_accuracy"]
        .index(
            max(
                history.history["val_accuracy"]
            )
        )
    )

    summary = {

        "selected_metric": "val_accuracy",

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
    config
):

    metadata = {

        "experiment_name":
            config.experiment_name,

        "model_name":
            config.model_name,

        "image_size":
            list(config.image_size),

        "batch_size":
            config.batch_size,

        "epochs":
            config.epochs,

        "learning_rate":
            config.learning_rate,

        "learning_rate_display":
            config.learning_rate_display,

        "seed":
            config.seed,

        "num_classes":
            config.num_classes,

        "trainable_backbone":
            config.trainable_backbone,

        "train_csv":
            config.train_csv.as_posix(),

        "val_csv":
            config.val_csv.as_posix(),

        "test_csv":
            config.test_csv.as_posix(),

        "train_csv_hash":
            calculate_file_hash(
                config.train_csv
            ),

        "val_csv_hash":
            calculate_file_hash(
                config.val_csv
            ),

        "test_csv_hash":
            calculate_file_hash(
                config.test_csv
            ),

        "dataset_hash":
            calculate_dataset_hash(
                config.train_csv,
                config.val_csv,
                config.test_csv
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