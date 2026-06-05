import json
import pandas as pd
from PIL import Image
from shutil import copy2
from pathlib import Path


RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")
REPORTS_DIR = Path("dataset/reports")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def standardize_dataset():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mapping_records = []

    total_images = 0
    class_names = []

    for class_dir in sorted(RAW_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_names.append(class_name)

        output_class_dir = PROCESSED_DIR / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_counter = 1

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            new_filename = (
                f"{class_name}_{image_counter:04d}.jpg"
            )

            new_path = output_class_dir / new_filename

            if image_path.suffix.lower() in [".jpg", ".jpeg"]:

                copy2(image_path, new_path)

            else:

                image = Image.open(image_path).convert("RGB")

                image.save(
                    new_path,
                    format="JPEG",
                    quality=95
                )

            mapping_records.append({
                "old_path": str(image_path),
                "new_path": str(new_path),
                "old_filename": image_path.name,
                "new_filename": new_filename,
                "old_extension": image_path.suffix.lower(),
                "new_extension": ".jpg"
            })

            image_counter += 1
            total_images += 1

    pd.DataFrame(mapping_records).to_csv(
        REPORTS_DIR / "file_mapping.csv",
        index=False
    )

    dataset_info = {
        "num_classes": len(class_names),
        "total_images": total_images,
        "image_format": "jpg"
    }

    with open(
        REPORTS_DIR / "dataset_info.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            dataset_info,
            f,
            indent=4
        )

    print("Dataset standardization completed.")

if __name__ == "__main__":
    standardize_dataset()