from src.data.loaders.load_image import (
    load_image
)

from src.preprocessing.config.preprocessing_config import (
    PreprocessingConfig
)


def load_visualization_image(
    image_path,
    image_size,
    color_mode
):

    preprocessing_config = (
        PreprocessingConfig(
            image_size=tuple(image_size),
            color_mode=color_mode
        )
    )

    image = load_image(
        image_path=image_path,
        preprocessing_config=preprocessing_config
    )

    return (
        image.numpy()
        .astype("uint8")
    )