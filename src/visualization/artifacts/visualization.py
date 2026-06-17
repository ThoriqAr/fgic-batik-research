from pathlib import Path
from dataclasses import dataclass
from matplotlib.figure import Figure


@dataclass(slots=True)
class Visualization:

    figure: Figure

    class_name: str

    npz_path: Path