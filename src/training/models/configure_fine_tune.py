import tensorflow as tf


def configure_model_for_fine_tune(
    model,
    train_config
):

    backbone = model.get_layer(
        "backbone"
    )

    backbone.trainable = False

    total_layers = len(
        backbone.layers
    )

    freeze_until = (
        total_layers -
        train_config.unfreeze_last_n_layers
    )

    for layer in backbone.layers[freeze_until:]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

        else:
            layer.trainable = True

    train_config.backbone_total_layers = (
        len(backbone.layers)
    )

    train_config.trainable_backbone_layers = sum(
        1
        for layer in backbone.layers
        if layer.trainable
    )

    trainable_count = sum(
        layer.trainable
        for layer in backbone.layers
    )

    print(
        f"Trainable backbone layers: "
        f"{trainable_count}/"
        f"{total_layers}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=train_config.learning_rate
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.05
        ),
        metrics=["accuracy"]
    )

    return model