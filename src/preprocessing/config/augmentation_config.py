from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    
    preset_name: str = "custom"

    rotation: float = 0.0

    width_shift: float = 0.0

    height_shift: float = 0.0

    zoom: float = 0.0

    brightness: tuple | None = None