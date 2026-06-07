import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50

def build_resnet50(config):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(
            config.image_size[0],
            config.image_size[1],
            3
        )
    )

    base_model.trainable = config.trainable_backbone

    inputs = tf.keras.Input(
        shape=(
            config.image_size[0],
            config.image_size[1],
            3
        )
    )

    x = base_model(
        inputs,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        config.num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model