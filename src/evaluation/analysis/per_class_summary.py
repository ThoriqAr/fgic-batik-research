import pandas as pd


def build_per_class_summary(
    dataframe
):
    
    summary = (
        dataframe
        .groupby(
            "ground_truth"
        )
        .agg(
            samples=(
                "ground_truth",
                "count"
            ),

            insertion_auc=(
                "insertion_auc",
                "mean"
            ),

            deletion_auc=(
                "deletion_auc",
                "mean"
            ),

            gaussian_noise=(
                "gaussian_noise",
                "mean"
            ),

            brightness=(
                "brightness",
                "mean"
            ),

            rotation=(
                "rotation",
                "mean"
            )
        )
        .reset_index()
    )

    return summary