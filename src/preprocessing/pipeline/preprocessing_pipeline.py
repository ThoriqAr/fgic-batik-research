from src.preprocessing.transforms.resize import (
    resize_image
)

from src.preprocessing.transforms.color import (
    apply_color_mode
)


def apply_preprocessing(
    image,
    preprocessing_config
):

    image = apply_color_mode(
        image,
        preprocessing_config.color_mode
    )

    image = resize_image(
        image,
        preprocessing_config.image_size
    )

    return image