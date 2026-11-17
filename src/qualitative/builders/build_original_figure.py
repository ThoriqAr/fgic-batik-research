import matplotlib.pyplot as plt

from src.visualization.loaders.image_loader import (
    load_visualization_image
)


def build_original_figure(
    attribution
):

    image = load_visualization_image(
        image_path=attribution["img_path"],
        image_size=attribution["image_size"],
        color_mode=attribution["color_mode"]
    )

    fig = plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image
    )

    plt.axis(
        "off"
    )

    plt.title(
        "Original Image"
    )

    plt.tight_layout()

    return fig