import tensorflow as tf


PREPROCESS_REGISTRY = {

    "mobilenetv2":
        tf.keras.applications.mobilenet_v2.preprocess_input,

    "resnet50":
        tf.keras.applications.resnet50.preprocess_input,
}


def get_preprocess_fn(
    model_name
):

    model_name = model_name.lower()

    if model_name not in PREPROCESS_REGISTRY:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return PREPROCESS_REGISTRY[
        model_name
    ]