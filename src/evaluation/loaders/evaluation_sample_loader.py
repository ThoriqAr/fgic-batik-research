from src.evaluation.loaders.evaluation_image_loader import (
    load_evaluation_image
)

from src.evaluation.loaders.evaluation_target_loader import (
    load_evaluation_target
)

from src.evaluation.loaders.evaluation_attribution_loader import (
    load_evaluation_attribution
)


def load_evaluation_sample(
    sample,
    model_context
):

    x_batch = (
        load_evaluation_image(
            sample,
            model_context
        )
    )

    y_batch = (
        load_evaluation_target(
            sample
        )
    )

    a_batch = (
        load_evaluation_attribution(
            sample
        )
    )

    return {

        "x_batch":
            x_batch,

        "y_batch":
            y_batch,

        "a_batch":
            a_batch
    }