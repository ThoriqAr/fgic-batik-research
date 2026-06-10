import tensorflow as tf
from pathlib import Path


def create_callbacks(artifact_dir):

    artifact_dir = Path(artifact_dir)

    checkpoint_path = (
        artifact_dir /
        "best_model.keras"
    )

    history_path = (
        artifact_dir /
        "history_training.csv"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            filename=str(history_path)
        )

    ]

    return callbacks