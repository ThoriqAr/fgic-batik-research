import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_max_similarity_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(dataframe["category"].unique()):

        subset = dataframe[
            dataframe["category"] == category
        ]

        plt.hist(
            subset["max_similarity"],
            bins=20,
            alpha=0.6,
            label=category
        )

    plt.xlabel("Maximum Similarity")
    plt.ylabel("Frequency")
    plt.title("Maximum Similarity Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "max_similarity_histogram.png",
        dpi=300
    )

    plt.close()


def plot_mean_topk_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(dataframe["category"].unique()):

        subset=dataframe[
            dataframe["category"]==category
        ]

        plt.hist(
            subset["mean_topk_similarity"],
            bins=20,
            alpha=0.6,
            label=category
        )

    plt.xlabel("Mean Top-5 Similarity")
    plt.ylabel("Frequency")
    plt.title("Mean Top-5 Similarity Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir/
        "mean_topk_similarity_histogram.png",
        dpi=300
    )

    plt.close()


def plot_std_topk_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(dataframe["category"].unique()):

        subset=dataframe[
            dataframe["category"]==category
        ]

        plt.hist(
            subset["std_topk_similarity"],
            bins=20,
            alpha=0.6,
            label=category
        )

    plt.xlabel("Std Top-5 Similarity")
    plt.ylabel("Frequency")
    plt.title("Std Top-5 Similarity Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir/
        "std_topk_similarity_histogram.png",
        dpi=300
    )

    plt.close()


def plot_agreement_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(
        dataframe["category"].unique()
    ):

        subset = dataframe[
            dataframe["category"] == category
        ]

        plt.hist(
            subset["agreement"],
            bins=10,
            alpha=0.6,
            label=category
        )

    plt.xlabel(
        "Agreement Ratio"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Agreement Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "agreement_histogram.png",
        dpi=300
    )

    plt.close()


def plot_unique_neighbor_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(dataframe["category"].unique()):

        subset=dataframe[
            dataframe["category"]==category
        ]

        values=subset[
            "unique_neighbor_labels"
        ]

        plt.hist(
            values,
            bins=np.arange(1,7)-0.5,
            alpha=0.6,
            label=category
        )

    plt.xticks([1,2,3,4,5])

    plt.xlabel("Unique Neighbor Labels")

    plt.ylabel("Frequency")

    plt.title("Unique Neighbor Labels")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir/
        "unique_neighbor_histogram.png",
        dpi=300
    )

    plt.close()


def plot_margin_histogram(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,5))

    for category in sorted(
        dataframe["category"].unique()
    ):

        subset = dataframe[
            dataframe["category"] == category
        ]

        plt.hist(
            subset["margin"],
            bins=20,
            alpha=0.6,
            label=category
        )

    plt.xlabel(
        "Similarity Margin"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Similarity Margin Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "margin_histogram.png",
        dpi=300
    )

    plt.close()


def plot_boxplots(
    dataframe,
    output_dir
):

    metrics=[

        "max_similarity",

        "agreement",

        "margin",

        "mean_topk_similarity",

        "std_topk_similarity",

        "unique_neighbor_labels"

    ]

    for metric in metrics:

        plt.figure(figsize=(6,5))

        dataframe.boxplot(
            column=metric,
            by="category"
        )

        plt.suptitle("")

        plt.title(metric)

        plt.tight_layout()

        plt.savefig(

            output_dir /

            f"{metric}_boxplot.png",

            dpi=300

        )

        plt.close()

  
def plot_similarity_agreement_scatter(
    dataframe,
    output_dir
):

    plt.figure(figsize=(7,6))

    markers = {

        "known":"o",

        "unknown":"^",

        "non_batik":"s"

    }

    for category in sorted(
        dataframe["category"].unique()
    ):

        subset = dataframe[
            dataframe["category"] == category
        ]

        plt.scatter(

            subset["max_similarity"],

            subset["agreement"],

            marker=markers[category],

            label=category,

            alpha=0.8

        )

    plt.xlabel(
        "Max Similarity"
    )

    plt.ylabel(
        "Agreement Ratio"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_dir /

        "similarity_scatter.png",

        dpi=300

    )

    plt.close()


def plot_similarity_margin_scatter(
    dataframe,
    output_dir
):

    plt.figure(figsize=(7,6))

    for category in sorted(dataframe["category"].unique()):

        subset=dataframe[
            dataframe["category"]==category
        ]

        plt.scatter(

            subset["max_similarity"],

            subset["margin"],

            alpha=0.8,

            label=category

        )

    plt.xlabel("Max Similarity")

    plt.ylabel("Margin")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir/
        "similarity_margin_scatter.png",
        dpi=300
    )

    plt.close()


def plot_similarity_topk_scatter(
    dataframe,
    output_dir
):

    plt.figure(figsize=(7,6))

    for category in sorted(dataframe["category"].unique()):

        subset=dataframe[
            dataframe["category"]==category
        ]

        plt.scatter(

            subset["mean_topk_similarity"],

            subset["std_topk_similarity"],

            alpha=0.8,

            label=category

        )

    plt.xlabel("Mean Top-5 Similarity")

    plt.ylabel("Std Top-5 Similarity")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir/
        "topk_similarity_scatter.png",
        dpi=300
    )

    plt.close()


def plot_similarity_cdf(
    dataframe,
    output_dir
):

    plt.figure(figsize=(8,6))

    for category in sorted(dataframe["category"].unique()):

        subset=np.sort(

            dataframe[
                dataframe["category"]==category
            ]["max_similarity"]

        )

        y=np.arange(

            1,

            len(subset)+1

        )/len(subset)

        plt.plot(

            subset,

            y,

            linewidth=2,

            label=category

        )

    plt.xlabel("Maximum Similarity")

    plt.ylabel("CDF")

    plt.title("CDF of Maximum Similarity")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_dir/

        "max_similarity_cdf.png",

        dpi=300

    )

    plt.close()


def generate_similarity_plots(
    dataframe,
    output_dir
):

    output_dir = output_dir / "plots"
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_max_similarity_histogram(
        dataframe,
        output_dir
    )

    plot_mean_topk_histogram(
        dataframe,
        output_dir
    )

    plot_std_topk_histogram(
        dataframe,
        output_dir
    )

    plot_margin_histogram(
        dataframe,
        output_dir
    )

    plot_agreement_histogram(
        dataframe,
        output_dir
    )

    plot_unique_neighbor_histogram(
        dataframe,
        output_dir
    )

    plot_boxplots(
        dataframe,
        output_dir
    )

    plot_similarity_margin_scatter(
        dataframe,
        output_dir
    )

    plot_similarity_agreement_scatter(
        dataframe,
        output_dir
    )

    plot_similarity_topk_scatter(
        dataframe,
        output_dir
    )

    plot_similarity_cdf(
        dataframe,
        output_dir
    )