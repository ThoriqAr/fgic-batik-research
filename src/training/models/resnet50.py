import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50

def build_resnet50(train_config):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(
            train_config.preprocessing_config.image_size[0],
            train_config.preprocessing_config.image_size[1],
            3
        )
    )

    base_model.trainable = train_config.trainable_backbone

    inputs = tf.keras.Input(
        shape=(
            train_config.preprocessing_config.image_size[0],
            train_config.preprocessing_config.image_size[1],
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
        train_config.num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=train_config.learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model