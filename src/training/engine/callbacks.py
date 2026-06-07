import tensorflow as tf
from pathlib import Path


def create_callbacks(output_dir):

    output_dir = Path(output_dir)

    checkpoint_path = (
        output_dir /
        "best_model.keras"
    )

    history_path = (
        output_dir /
        "training_artifacts" /
        "history_training.csv"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            filename=str(history_path)
        )

    ]

    return callbacks