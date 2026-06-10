import json

def load_training_metadata(metadata_path):

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)