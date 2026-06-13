from dataclasses import dataclass


@dataclass
class LimeConfig:

    num_samples: int = 1000

    n_segments: int = 150

    compactness: int = 5

    sigma: int = 1

    seed: int = 42