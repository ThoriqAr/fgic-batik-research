from datetime import datetime

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from src.utils.path_config import (
    ARTIFACTS_DIR,
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

from src.training.models.mobilenetv2 import (
    build_mobilenetv2
)

from src.training.models.resnet50 import (
    build_resnet50
)


def run_training(train_config):

    set_global_seed(
        train_config.seed
    )

    experiment_id = (
        f"{train_config.experiment_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = (
        ARTIFACTS_DIR /
        "models" /
        experiment_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    training_dir = (
        output_dir /
        "training_artifacts"
    )

    training_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n[1/3] Build model...")

    if train_config.model_name == "mobilenetv2":

        preprocessing_fn = (
            mobilenet_preprocess
        )

        model = build_mobilenetv2(
            train_config
        )

    elif train_config.model_name == "resnet50":

        preprocessing_fn = (
            resnet_preprocess
        )

        model = build_resnet50(
            train_config
        )

    else:

        raise ValueError(
            f"Unknown model: {train_config.model_name}"
        )
    
    print(
        f"Model created: {train_config.model_name}"
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
        output_dir
    )

    print("Training completed.")

    print("\n[3/3] Save model & metadata...")

    model.save(
        output_dir /
        "model.keras"
    )

    history_csv = (
        training_dir /
        "history_training.csv"
    )

    save_metadata(
        training_dir /
        "metadata.json",
        train_config
    )

    save_best_epoch(
        history,
        training_dir /
        "best_epoch.json"
    )

    save_learning_curves(
        history_csv,
        training_dir /
        "learning_curves.png"
    )

    format_history_csv(
        history_csv,
        training_dir /
        "history_formatted.csv"
    )

    print("Model & metadata saved.")

    print(
        f"\nTraining Completed : {experiment_id}"
    )