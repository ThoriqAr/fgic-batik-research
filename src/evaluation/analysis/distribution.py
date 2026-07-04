import matplotlib.pyplot as plt


def plot_metric_distribution(
    dataframe,
    metric_name,
    output_dir=None,
    bins=20
):

    values = dataframe[metric_name]

    mean = values.mean()
    std = values.std()
    min_value = values.min()
    max_value = values.max()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={
            "height_ratios": [4, 1]
        }
    )

    histogram_ax = axes[0]
    boxplot_ax = axes[1]

    histogram_ax.set_xlim(0, 1)
    boxplot_ax.set_xlim(0, 1)

    histogram_ax.hist(
        values,
        bins=bins
    )

    histogram_ax.axvline(
        mean,
        linestyle="--",
        label=f"Mean = {mean:.3f}"
    )

    histogram_ax.text(
        0.98,
        0.95,
        (
            f"n = {len(values)}\n"
            f"mean = {mean:.3f}\n"
            f"std = {std:.3f}\n"
            f"min = {min_value:.3f}\n"
            f"max = {max_value:.3f}"
        ),
        transform=histogram_ax.transAxes,
        ha="right",
        va="top"
    )

    histogram_ax.set_title(
        f"{metric_name} Distribution"
    )

    histogram_ax.set_ylabel(
        "Frequency"
    )

    histogram_ax.legend()

    boxplot_ax.boxplot(
        values,
        vert=False
    )

    boxplot_ax.set_xlabel(
        metric_name
    )

    plt.tight_layout()

    if output_dir:

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_dir /
            f"{metric_name}.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    plt.close()


def plot_all_metrics(
    dataframe,
    output_dir=None
):

    metrics = [
        "insertion_auc",
        "deletion_auc",
        "gaussian_noise",
        "brightness",
        "rotation"
    ]

    for metric in metrics:

        plot_metric_distribution(
            dataframe=dataframe,
            metric_name=metric,
            output_dir=output_dir
        )