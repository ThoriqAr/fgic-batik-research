import pandas as pd


def save_error_log(
    error_records,
    output_path
):

    df = pd.DataFrame(
        error_records,
        columns=[
            "attribution_id",
            "error"
        ]
    )

    df.to_csv(
        output_path,
        index=False
    )

    return df