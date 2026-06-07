import json

from src.utils.path_config import METADATA_DIR


def load_class_mapping():

    mapping_path = (
        METADATA_DIR /
        "class_mapping.json"
    )

    with open(
        mapping_path,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    return mapping


def load_id_to_class_mapping():

    mapping = load_class_mapping()

    return {
        class_id: class_name
        for class_name, class_id
        in mapping.items()
    }


def get_class_names():

    id_to_class = (
        load_id_to_class_mapping()
    )

    return [
        id_to_class[i]
        for i in sorted(id_to_class)
    ]