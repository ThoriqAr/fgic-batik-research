import pandas as pd


def add_consensus_statistics(
    comparison_df
):

    metric_columns = [

        column

        for column in comparison_df.columns

        if column != "ground_truth"
    ]

    result = comparison_df.copy()

    result["mean_score"] = (
        result[
            metric_columns
        ]
        .mean(axis=1)
    )

    result["std_between_models"] = (
        result[
            metric_columns
        ]
        .std(axis=1)
    )

    return result


def compare_per_class_metric(
    summaries,
    metric_name
):

    merged_df = None

    for (
        experiment_name,
        dataframe
    ) in summaries.items():

        current = dataframe[
            [
                "ground_truth",
                metric_name
            ]
        ].rename(
            columns={
                metric_name:
                    experiment_name
            }
        )

        if merged_df is None:

            merged_df = current

        else:

            merged_df = (
                merged_df.merge(
                    current,
                    on="ground_truth",
                    how="outer"
                )
            )

    return merged_df


def rank_difficult_classes(
    comparison_df
):

    result = add_consensus_statistics(
        comparison_df
    )

    return (

        result

        .sort_values(
            "mean_score"
        )

        .reset_index(
            drop=True
        )

    )


def rank_disagreement_classes(
    comparison_df
):

    result = add_consensus_statistics(
        comparison_df
    )

    return (

        result

        .sort_values(

            "std_between_models",

            ascending=False
        )

        .reset_index(
            drop=True
        )

    )


def rank_consensus_difficult_classes(
    comparison_df
):

    result = add_consensus_statistics(
        comparison_df
    )

    return (

        result

        .sort_values(

            by=[
                "mean_score",
                "std_between_models"
            ],

            ascending=[
                True,
                True
            ]
        )

        .reset_index(
            drop=True
        )

    )


def rank_consensus_easy_classes(
    comparison_df
):

    result = add_consensus_statistics(
        comparison_df
    )

    return (

        result

        .sort_values(

            by=[
                "mean_score",
                "std_between_models"
            ],

            ascending=[
                False,
                True
            ]
        )

        .reset_index(
            drop=True
        )

    )