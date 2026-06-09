from src.preprocessing.config.augmentation_config import (
    AugmentationConfig
)


AUGMENTATION_PRESETS = {

    "none": AugmentationConfig(
        preset_name="none"
    ),

    "light": AugmentationConfig(
        preset_name="light",
        rotation=10,
        width_shift=0.05,
        height_shift=0.05,
        zoom=0.10,
        brightness=(0.9, 1.1)
    ),

    "medium": AugmentationConfig(
        preset_name="medium",
        rotation=20,
        width_shift=0.10,
        height_shift=0.10,
        zoom=0.20,
        brightness=(0.8, 1.2)
    ),

    "strong": AugmentationConfig(
        preset_name="strong",
        rotation=30,
        width_shift=0.15,
        height_shift=0.15,
        zoom=0.30,
        brightness=(0.7, 1.3)
    )
}