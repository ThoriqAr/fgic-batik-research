import numpy as np


def apply_brightness(
    image,
    factor=1.2
):

    perturbed = (

        image.astype(np.float32)
        * factor

    )

    perturbed = np.clip(

        perturbed,
        0,
        255
    )

    return perturbed.astype(
        np.uint8
    )