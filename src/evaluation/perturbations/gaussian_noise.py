import numpy as np


def apply_gaussian_noise(
    image,
    sigma=10.0,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(

        loc=0.0,

        scale=sigma,

        size=image.shape
    )

    perturbed = (

        image.astype(np.float32)
        + noise

    )

    perturbed = np.clip(

        perturbed,
        0,
        255
    )

    return perturbed.astype(
        np.uint8
    )