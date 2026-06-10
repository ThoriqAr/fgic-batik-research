import tensorflow as tf
from pathlib import Path


from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from src.training.models.configure_fine_tune import (
    configure_model_for_fine_tune
)

from src.training.data.dataset import (
    create_training_datasets
)

from src.training.engine.trainer import (
    train_model
)

from src.training.artifacts.training_artifacts import (
    save_best_epoch,
    save_metadata,
    save_learning_curves,
    format_history_csv
)

from src.utils.seed import (
    set_global_seed
)


def run_fine_tuning(train_config):
  
    set_global_seed(
        train_config.seed
    )

    output_dir = (train_config.fine_tune_from.parent.parent)

    fine_tune_dir = (
        output_dir /
        f"fine_tune_stage_{train_config.fine_tune_stage}"
    )

    fine_tune_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n[1/3] Load model...")

    model = tf.keras.models.load_model(
        train_config.fine_tune_from
    )

    model = configure_model_for_fine_tune(
        model,
        train_config
    )
    
    print(
        f"Model loaded: {train_config.model_name}"
    )

    if train_config.model_name == "mobilenetv2":

        preprocessing_fn = mobilenet_preprocess

    elif train_config.model_name == "resnet50":

        preprocessing_fn = resnet_preprocess

    else:

        raise ValueError(
            f"Unknown model: {train_config.model_name}"
        )

    train_ds, val_ds = (
        create_training_datasets(
            train_config,
            preprocessing_fn
        )
    )

    print("\n[2/3] Start training...")

    history = train_model(
        model,
        train_ds,
        val_ds,
        train_config,
        fine_tune_dir
    )

    print("Training completed.")

    print("\n[3/3] Save metadata...")

    history_csv = (
        fine_tune_dir /
        "history_training.csv"
    )

    save_metadata(
        fine_tune_dir /
        "metadata.json",
        train_config
    )

    save_best_epoch(
        history,
        fine_tune_dir /
        "best_epoch.json"
    )

    save_learning_curves(
        history_csv,
        fine_tune_dir /
        "learning_curves.png"
    )

    format_history_csv(
        history_csv,
        fine_tune_dir /
        "history_formatted.csv"
    )

    print("Model & metadata saved.")

    print(
        f"\nTraining Completed : {output_dir.name}/{fine_tune_dir.name}"
    )