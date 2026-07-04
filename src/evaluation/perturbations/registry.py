from src.evaluation.perturbations.gaussian_noise import (
    apply_gaussian_noise
)

from src.evaluation.perturbations.brightness import (
    apply_brightness
)

from src.evaluation.perturbations.rotation import (
    apply_rotation
)


PERTURBATION_REGISTRY = {

    "gaussian_noise":
        apply_gaussian_noise,

    "brightness":
        apply_brightness,

    "rotation":
        apply_rotation
}


def apply_perturbation(
    image,
    perturbation_name,
    **kwargs
):

    if perturbation_name not in (
        PERTURBATION_REGISTRY
    ):

        raise ValueError(
            f"Unknown perturbation: "
            f"{perturbation_name}"
        )

    return (
        PERTURBATION_REGISTRY[
            perturbation_name
        ](
            image,
            **kwargs
        )
    )