import pandas as pd

from src.evaluation.analysis.load_summary import (
    load_summary
)


def extract_summary_metrics(
    summary
):

    return {

        "insertion_auc":
            summary["faithfulness"]
                   ["insertion_auc"]
                   ["mean"],

        "deletion_auc":
            summary["faithfulness"]
                   ["deletion_auc"]
                   ["mean"],

        "gaussian_noise":
            summary["stability"]
                   ["gaussian_noise"]
                   ["mean"],

        "brightness":
            summary["stability"]
                   ["brightness"]
                   ["mean"],

        "rotation":
            summary["stability"]
                   ["rotation"]
                   ["mean"]
    }


def compare_experiments(
    experiments
):

    rows = []

    for (
        experiment_name,
        summary_path
    ) in experiments.items():

        summary = (
            load_summary(
                summary_path
            )
        )

        metrics = (
            extract_summary_metrics(
                summary
            )
        )

        metrics[
            "experiment"
        ] = experiment_name

        rows.append(
            metrics
        )

    df = pd.DataFrame(
        rows
    )

    columns = [
        "experiment",
        "insertion_auc",
        "deletion_auc",
        "gaussian_noise",
        "brightness",
        "rotation"
    ]

    df = df[
        columns
    ]

    return df