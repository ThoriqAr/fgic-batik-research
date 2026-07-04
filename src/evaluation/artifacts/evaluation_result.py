import pandas as pd
from dataclasses import dataclass


@dataclass
class EvaluationResult:

    attribution_id: str

    ground_truth: str

    pred_label: str

    confidence: float

    insertion_auc: float

    deletion_auc: float

    stability_scores: dict[str, float]


def build_evaluation_result(
    sample,
    attribution,
    faithfulness_result,
    stability_result
):

    return EvaluationResult(
        attribution_id=sample.attribution_id,
        ground_truth=attribution["ground_truth"],
        pred_label=attribution["pred_label"],
        confidence=float(attribution["confidence"]),
        insertion_auc=float(faithfulness_result["insertion_auc"]),
        deletion_auc=float(faithfulness_result["deletion_auc"]),
        stability_scores=stability_result
    )


def save_results_csv(
    results,
    output_path
):

    rows = []

    for result in results:

        row = {
            "attribution_id":result.attribution_id,
            "ground_truth":result.ground_truth,
            "pred_label":result.pred_label,
            "confidence":result.confidence,
            "insertion_auc":result.insertion_auc,
            "deletion_auc":result.deletion_auc,
            "gaussian_noise":result.stability_scores["gaussian_noise"],
            "brightness":result.stability_scores["brightness"],
            "rotation":result.stability_scores["rotation"]
        }

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df