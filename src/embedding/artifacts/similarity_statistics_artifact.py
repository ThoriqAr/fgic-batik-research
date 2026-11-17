import json


def save_similarity_summary(
    summary,
    output_dir
):
    
    output_dir = output_dir / "analysis"
    
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )