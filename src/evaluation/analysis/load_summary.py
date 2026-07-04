import json


def load_summary(
    summary_path
):

    with open(
        summary_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)