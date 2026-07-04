import pandas as pd


def load_manifest(attribution_dir):

    manifest_path = (
        attribution_dir.parent /
        "manifest.csv"
    )

    return pd.read_csv(manifest_path)


def filter_manifest(
    manifest,
    classes=None,
    filenames=None,
    attribution_ids=None,
    max_samples=None
):

    df = manifest.copy()

    if attribution_ids is not None:
        df = df[
            df["attribution_id"].isin(attribution_ids)
        ]

    elif filenames is not None:
        df = df[
            df["file_name"].isin(filenames)
        ]

    elif classes is not None:
        df = df[
            df["ground_truth"].isin(classes)
        ]

    if max_samples is not None:
        df = (
            df.groupby("ground_truth")
              .head(max_samples)
        )

    return df