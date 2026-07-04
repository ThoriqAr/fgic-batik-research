import numpy as np

from src.evaluation.faithfulness.utils import (
    predict_target_confidence,
    build_black_image
)

from src.evaluation.faithfulness.auc import (
    compute_auc
)


def build_deletion_curve(
    image,
    ranking,
    model,
    target_class,
    preprocess_fn,
    num_steps=20
):
  
    image = image.copy()

    black_image = (
        build_black_image(
            image
        )
    )

    total_pixels = (
        ranking.size
    )

    pixel_schedule = np.linspace(
        0,
        total_pixels,
        num_steps + 1,
        dtype=int
    )

    current_image = (
        image.copy()
    )

    x_values = [0.0]

    y_values = [

        predict_target_confidence(
            model=model,
            image=image,
            target_class=target_class,
            preprocess_fn=preprocess_fn
        )

    ]

    for step in range(
        1,
        num_steps + 1
    ):
        
        end_idx = pixel_schedule[step]

        selected_pixels = (
            ranking[:end_idx]
        )

        rows, cols = np.unravel_index(
            selected_pixels,
            image.shape[:2]
        )

        current_image[
            rows,
            cols
        ] = black_image[
            rows,
            cols
        ]

        confidence = (
            predict_target_confidence(
                model=model,
                image=current_image,
                target_class=target_class,
                preprocess_fn=preprocess_fn
            )
        )

        x_values.append(
            end_idx /
            total_pixels
        )

        y_values.append(
            confidence
        )

    return {

        "x":
            np.array(
                x_values,
                dtype=np.float32
            ),

        "y":
            np.array(
                y_values,
                dtype=np.float32
            )
    }


def run_deletion_test(
    image,
    ranking,
    model,
    target_class,
    preprocess_fn,
    num_steps=20
):

    curve = (
        build_deletion_curve(
            image=image,
            ranking=ranking,
            model=model,
            target_class=target_class,
            preprocess_fn=preprocess_fn,
            num_steps=num_steps
        )
    )

    auc_score = (
        compute_auc(
            curve["x"],
            curve["y"]
        )
    )

    return {
        "auc":auc_score,
        "curve":curve
    }