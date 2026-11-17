import numpy as np
import pandas as pd


def summarize_similarity_statistics(
    dataframe: pd.DataFrame
):

    summaries = {}

    categories = sorted(
        dataframe["category"].unique()
    )

    for category in categories:

        subset = dataframe[
            dataframe["category"] == category
        ]

        similarities = subset[
            "max_similarity"
        ].to_numpy()

        agreements = subset[
            "agreement"
        ].to_numpy()

        margins = subset[
            "margin"
        ].to_numpy()

        mean_topk = subset[
            "mean_topk_similarity"
        ].to_numpy()

        std_topk = subset[
            "std_topk_similarity"
        ].to_numpy()

        unique_neighbor = subset[
            "unique_neighbor_labels"
        ].to_numpy()

        consistency = subset[
            "label_consistency"
        ].astype(int).to_numpy()

        summaries[category] = {

            "count":int(len(similarities)),

            "max_similarity": {

                "mean":float(np.mean(similarities)),
                "std":float(np.std(similarities)),
                "median":float(np.median(similarities)),
                "min":float(np.min(similarities)),
                "max":float(np.max(similarities)),
                "q1":float(np.percentile(similarities, 25)),
                "q3":float(np.percentile(similarities, 75)),
                "p95":float(np.percentile(similarities, 95)),
                "p99":float(np.percentile(similarities, 99))

            },

            "agreement_ratio": {

                "mean":float(np.mean(agreements)),
                "std":float(np.std(agreements)),
                "median":float(np.median(agreements)),
                "min":float(np.min(agreements)),
                "max":float(np.max(agreements))

            },

            "similarity_margin": {

                "mean":float(np.mean(margins)),
                "std":float(np.std(margins)),
                "median":float(np.median(margins)),
                "min":float(np.min(margins)),
                "max":float(np.max(margins))

            },

            "mean_topk_similarity": {

                "mean": float(np.mean(mean_topk)),
                "std": float(np.std(mean_topk)),
                "median": float(np.median(mean_topk)),
                "min": float(np.min(mean_topk)),
                "max": float(np.max(mean_topk))

            },

            "std_topk_similarity": {

                "mean": float(np.mean(std_topk)),
                "std": float(np.std(std_topk)),
                "median": float(np.median(std_topk)),
                "min": float(np.min(std_topk)),
                "max": float(np.max(std_topk))

            },

            "unique_neighbor_labels": {

                "mean": float(np.mean(unique_neighbor)),
                "std": float(np.std(unique_neighbor)),
                "min": int(np.min(unique_neighbor)),
                "max": int(np.max(unique_neighbor))

            },

            "label_consistency": {

                "mean": float(np.mean(consistency))

            }

        }

    return summaries