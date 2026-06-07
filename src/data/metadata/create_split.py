import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.path_config import (
    PROCESSED_DIR,
    METADATA_DIR
)

def create_split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):

    with open(
        METADATA_DIR / "class_mapping.json",
        "r",
        encoding="utf-8"
    ) as f:
        class_mapping = json.load(f)

    records = []

    for class_name, class_id in class_mapping.items():

        class_dir = PROCESSED_DIR / class_name

        for image_path in class_dir.glob("*.jpg"):

            records.append({
                "filepath": str(
                    image_path.relative_to(PROCESSED_DIR)
                ).replace("\\", "/"),
                "label": class_name,
                "class_id": class_id
            })

    df = pd.DataFrame(records)

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["class_id"],
        random_state=random_state
    )

    val_size_adjusted = (
        val_ratio /
        (val_ratio + test_ratio)
    )

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size_adjusted,
        stratify=temp_df["class_id"],
        random_state=random_state
    )

    train_df.to_csv(
        METADATA_DIR / "train.csv",
        index=False
    )

    val_df.to_csv(
        METADATA_DIR / "val.csv",
        index=False
    )

    test_df.to_csv(
        METADATA_DIR / "test.csv",
        index=False
    )

    print("Dataset split completed.")

if __name__ == "__main__":
    create_split()