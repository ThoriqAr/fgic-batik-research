from src.preprocessing.augmentations.presets import (
    AUGMENTATION_PRESETS
)


def get_augmentation_config(
    preset_name: str
):

    if preset_name not in AUGMENTATION_PRESETS:

        raise ValueError(
            f"Unknown augmentation preset: {preset_name}"
        )

    return AUGMENTATION_PRESETS[preset_name]