from dataclasses import dataclass

from matplotlib.figure import Figure

from src.qualitative.models.qualitative_sample import (
    QualitativeSample
)


@dataclass(slots=True)
class QualitativeVisualization:

    sample: QualitativeSample

    original: Figure

    explanation_a: Figure

    explanation_b: Figure

    explanation_c: Figure

    explanation_d: Figure