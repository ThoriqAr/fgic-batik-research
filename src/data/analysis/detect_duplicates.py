import json
import pandas as pd

from src.utils.calculate import (
    calculate_md5
)

from src.utils.path_config import (
    PROCESSED_DIR,
    REPORTS_DIR
)


def detect_duplicates():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    hash_map = {}

    image_paths = sorted(
        PROCESSED_DIR.rglob("*.jpg")
    )

    for image_path in image_paths:

        file_hash = calculate_md5(
            image_path
        )

        hash_map.setdefault(
            file_hash,
            []
        ).append(
            image_path
        )

    duplicate_records = []

    duplicate_groups = 0

    duplicate_images = 0

    for file_hash, paths in hash_map.items():

        if len(paths) <= 1:
            continue

        duplicate_groups += 1

        duplicate_images += len(paths)

        for path in paths:

            duplicate_records.append({
                "hash": file_hash,
                "filepath": str(
                    path.relative_to(
                        PROCESSED_DIR
                    )
                )
            })

    duplicate_df = pd.DataFrame(
        duplicate_records
    )

    duplicate_df.to_csv(
        REPORTS_DIR / "duplicate_report.csv",
        index=False
    )

    summary = {
        "total_images": len(
            image_paths
        ),
        "duplicate_groups": duplicate_groups,
        "duplicate_images": duplicate_images,
        "has_duplicates": duplicate_groups > 0
    }

    with open(
        REPORTS_DIR
        / "duplicate_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    return summary


def main():

    summary = detect_duplicates()

    print(
        json.dumps(
            summary,
            indent=4
        )
    )


if __name__ == "__main__":
    main()