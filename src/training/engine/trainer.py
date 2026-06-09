from pathlib import Path

from src.training.engine.callbacks import (
    create_callbacks
)


def train_model(
    model,
    train_ds,
    val_ds,
    train_config,
    output_dir
):

    output_dir = Path(output_dir)

    callbacks = create_callbacks(
        output_dir
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_config.epochs,
        callbacks=callbacks
    )

    return history