import tensorflow as tf


def get_last_conv_layer(
    model,
    backbone_name="backbone"
):

    backbone = model.get_layer(
        backbone_name
    )

    for layer in reversed(backbone.layers):

        try:

            if len(layer.output.shape) == 4:

                return layer.name

        except Exception:

            continue

    raise ValueError(
        "No convolution layer found."
    )