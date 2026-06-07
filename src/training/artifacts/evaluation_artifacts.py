import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_report(
    report,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


def save_confusion_matrix(
    cm,
    output_path
):

    pd.DataFrame(cm).to_csv(
        output_path,
        index=False
    )


def save_confusion_matrix_plot(
    cm,
    class_names,
    output_path
):

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()