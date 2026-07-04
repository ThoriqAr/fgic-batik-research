import pandas as pd
from dataclasses import dataclass

from src.evaluation.artifacts.attribution_sample import (
    AttributionSample
)


@dataclass
class EvaluationManifest:

    dataframe: pd.DataFrame

    @property
    def total_samples(self):

        return len(
            self.dataframe
        )
    
    def get_samples(self):

        samples = []

        for row in self.dataframe.itertuples():

            samples.append(
                AttributionSample(
                    attribution_id=row.attribution_id,
                    attribution_path=row.attribution_path,
                    image_path=row.img_path,
                    ground_truth=row.ground_truth,
                    pred_label=row.pred_label,
                    pred_index=row.pred_index,
                    confidence=row.confidence
                )
            )

        return samples