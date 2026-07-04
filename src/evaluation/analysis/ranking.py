import matplotlib.pyplot as plt


def get_top_classes(
    dataframe,
    metric,
    top_n=5
):

    return (
        dataframe
        .sort_values(
            metric,
            ascending=False
        )
        .head(top_n)
    )


def get_bottom_classes(
    dataframe,
    metric,
    top_n=5
):

    return (
        dataframe
        .sort_values(
            metric,
            ascending=True
        )
        .head(top_n)
    )


def plot_class_ranking(
    dataframe,
    metric,
    top_n=10
):

    ranking = (
        dataframe
        .sort_values(
            metric,
            ascending=False
        )
        .head(top_n)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        ranking["ground_truth"],
        ranking[metric]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        metric
    )

    plt.title(
        f"Top {top_n} Classes - {metric}"
    )

    plt.tight_layout()

    plt.show()