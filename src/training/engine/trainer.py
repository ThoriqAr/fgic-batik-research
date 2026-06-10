from pathlib import Path

from src.training.engine.callbacks import (
    create_callbacks
)


def train_model(
    model,
    train_ds,
    val_ds,
    train_config,
    artifact_dir
):

    artifact_dir = Path(artifact_dir)

    callbacks = create_callbacks(
        artifact_dir
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_config.epochs,
        callbacks=callbacks
    )

    return history