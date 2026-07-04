import pandas as pd


def load_evaluation_results(
    results_csv
):

    return pd.read_csv(
        results_csv
    )