import numpy as np


def prepare_x_batch(
    x_batch
):

    return x_batch.astype(
        np.float32
    )


def prepare_y_batch(
    y_batch
):

    return y_batch.astype(
        np.int64
    )


def prepare_a_batch(
    a_batch
):

    if a_batch.ndim == 3:

        a_batch = np.expand_dims(
            a_batch,
            axis=-1
        )

    return a_batch.astype(
        np.float32
    )


def build_quantus_input(
    sample_data
):

    return {

        "x_batch":
            prepare_x_batch(
                sample_data[
                    "x_batch"
                ]
            ),

        "y_batch":
            prepare_y_batch(
                sample_data[
                    "y_batch"
                ]
            ),

        "a_batch":
            prepare_a_batch(
                sample_data[
                    "a_batch"
                ]
            )
    }