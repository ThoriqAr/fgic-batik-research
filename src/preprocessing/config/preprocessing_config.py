from dataclasses import dataclass


@dataclass
class PreprocessingConfig:

    image_size: tuple = (224, 224)

    color_mode: str = "rgb"