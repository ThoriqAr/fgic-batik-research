import json

from src.utils.path_config import (
    PROCESSED_DIR,
    METADATA_DIR
)

OUTPUT_PATH = METADATA_DIR / "class_mapping.json"


def generate_class_mapping():

    classes = sorted([
        p.name
        for p in PROCESSED_DIR.iterdir()
        if p.is_dir()
    ])

    class_mapping = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            class_mapping,
            f,
            indent=4
        )

    return class_mapping

if __name__ == "__main__":
    generate_class_mapping()