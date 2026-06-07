import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.path_config import (
    RAW_DIR,
    PROCESSED_DIR,
    METRICS_DIR
)

OUTPUT_DIR = METRICS_DIR / "eda"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def analyze_class_distribution():

    records = []

    for class_dir in sorted(PROCESSED_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        count = len(list(class_dir.glob("*.jpg")))

        records.append({
            "class_name": class_dir.name,
            "image_count": count
        })

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_DIR / "class_distribution.csv",
        index=False
    )

    summary = {
        "num_classes": int(len(df)),
        "total_images": int(df["image_count"].sum()),
        "mean_images_per_class": float(df["image_count"].mean()),
        "median_images_per_class": float(df["image_count"].median()),
        "max_images_per_class": int(df["image_count"].max()),
        "min_images_per_class": int(df["image_count"].min())
    }

    return df, summary


def analyze_image_properties():

    records = []

    image_paths = list(
        PROCESSED_DIR.rglob("*.jpg")
    )

    for image_path in image_paths:

        with Image.open(image_path) as img:

            width, height = img.size

            records.append({
                "filepath": str(
                    image_path.relative_to(
                        PROCESSED_DIR
                    )
                ),
                "width": width,
                "height": height,
                "aspect_ratio": width / height
            })

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_DIR / "image_properties.csv",
        index=False
    )

    summary = {
        "min_width": int(df["width"].min()),
        "max_width": int(df["width"].max()),
        "mean_width": float(df["width"].mean()),
        "min_height": int(df["height"].min()),
        "max_height": int(df["height"].max()),
        "mean_height": float(df["height"].mean())
    }

    return df, summary


def validate_dataset_migration():

    raw_count = 0

    for ext in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff",
        "*.webp"
    ]:

        raw_count += len(
            list(
                RAW_DIR.rglob(ext)
            )
        )

    processed_count = len(
        list(
            PROCESSED_DIR.rglob("*.jpg")
        )
    )

    summary = {
        "raw_images": raw_count,
        "processed_images": processed_count,
        "is_match": raw_count == processed_count
    }

    return summary


def save_dataset_summary(
    class_summary,
    image_summary,
    migration_summary
):

    summary = {
        "migration_validation": migration_summary,
        "class_distribution": class_summary,
        "image_properties": image_summary
    }

    with open(
        OUTPUT_DIR / "dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


def visualize_class_samples(
    samples_per_class=5,
    save_figure=True
):

    class_dirs = sorted([
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir()
    ])

    n_classes = len(class_dirs)

    fig, axes = plt.subplots(
        n_classes,
        samples_per_class,
        figsize=(
            samples_per_class * 3,
            n_classes * 3
        )
    )

    if n_classes == 1:
        axes = [axes]

    for row_idx, class_dir in enumerate(class_dirs):

        image_paths = list(
            class_dir.glob("*.jpg")
        )

        selected_images = random.sample(
            image_paths,
            min(
                samples_per_class,
                len(image_paths)
            )
        )

        for col_idx in range(samples_per_class):

            ax = axes[row_idx][col_idx]

            ax.axis("off")

            if col_idx >= len(selected_images):
                continue

            img = Image.open(
                selected_images[col_idx]
            )

            ax.imshow(img)

            if col_idx == 0:

                ax.set_title(
                    class_dir.name,
                    fontsize=8,
                    loc="left"
                )

    plt.tight_layout()

    if save_figure:

        output_path = (
            OUTPUT_DIR
            / "class_samples.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Saved: {output_path}"
        )

    plt.show()


def main():
    
    print(
        "\n[1/4] Validate dataset migration..."
    )

    migration_summary = (
        validate_dataset_migration()
    )


    print(
        "\n[2/4] Analyze class distribution..."
    )

    _, class_summary = (
        analyze_class_distribution()
    )

    print(
        "\n[3/4] Analyze image properties..."
    )

    _, image_summary = (
        analyze_image_properties()
    )

    print(
        "\n[4/4] Save dataset summary..."
    )

    save_dataset_summary(
        class_summary,
        image_summary,
        migration_summary
    )

    print(
        "\nEDA completed."
    )


if __name__ == "__main__":
    main()