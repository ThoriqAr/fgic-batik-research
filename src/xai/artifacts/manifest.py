import pandas as pd
from pathlib import Path


def create_manifest_record(
    attribution_id,
    image_path,
    ground_truth,
    pred_label,
    pred_index,
    confidence,
    attribution_path
):

    return {

        "attribution_id":
            attribution_id,

        "file_name":
            Path(image_path).name,

        "ground_truth":
            ground_truth,

        "pred_label":
            pred_label,

        "pred_index":
            pred_index,

        "confidence":
            confidence,

        "img_path":
            image_path.as_posix(),

        "attribution_path":
            attribution_path.as_posix()
    }


def save_manifest(
    records,
    output_path
):

    df = pd.DataFrame(records)

    df.to_csv(
        output_path,
        index=False
    )

    return df