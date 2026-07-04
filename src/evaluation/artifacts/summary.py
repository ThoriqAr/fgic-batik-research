import json
import numpy as np


def build_summary(
    results
):
  
    insertion_scores = [
        result.insertion_auc
        for result in results
    ]

    deletion_scores = [
        result.deletion_auc
        for result in results
    ]

    gaussian_scores = [
        result.stability_scores["gaussian_noise"]
        for result in results
    ]

    brightness_scores = [
        result.stability_scores["brightness"]
        for result in results
    ]

    rotation_scores = [
        result.stability_scores["rotation"]
        for result in results
    ]

    return {

        "total_samples":len(results),

        "faithfulness": {

            "insertion_auc": {
                "mean":float(np.mean(insertion_scores)),
                "std":float(np.std(insertion_scores)),
                "min":float(np.min(insertion_scores)),
                "max":float(np.max(insertion_scores))
            },

            "deletion_auc": {
                "mean":float(np.mean(deletion_scores)),
                "std":float(np.std(deletion_scores)),
                "min":float(np.min(deletion_scores)),
                "max":float(np.max(deletion_scores))
            }
        },

        "stability": {

            "gaussian_noise": {
                "mean":float(np.mean(gaussian_scores)),
                "std":float(np.std(gaussian_scores)),
                "min":float(np.min(gaussian_scores)),
                "max":float(np.max(gaussian_scores))
            },

            "brightness": {
                "mean":float(np.mean(brightness_scores)),
                "std":float(np.std(brightness_scores)),
                "min":float(np.min(brightness_scores)),
                "max":float(np.max(brightness_scores))
            },

            "rotation": {
                "mean":float(np.mean(rotation_scores)),
                "std":float(np.std(rotation_scores)),
                "min":float(np.min(rotation_scores)),
                "max":float(np.max(rotation_scores))
            }
        }
    }


def save_summary(
    summary,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )