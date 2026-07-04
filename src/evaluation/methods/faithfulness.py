from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.evaluation.loaders.evaluation_heatmap_loader import (
    load_evaluation_heatmap
)

from src.evaluation.faithfulness.ranking import (
    build_pixel_ranking
)

from src.evaluation.faithfulness.deletion import (
    run_deletion_test
)

from src.evaluation.faithfulness.insertion import (
    run_insertion_test
)


def evaluate_faithfulness(
    sample,
    original_image,
    model_context,
    num_steps=20
):
  
    attribution = (
        load_attribution(
            sample.attribution_path
        )
    )

    target_class = int(
        attribution[
            "pred_index"
        ]
    )

    heatmap = (
        load_evaluation_heatmap(
            sample
        )
    )

    ranking = (
        build_pixel_ranking(
            heatmap
        )
    )

    deletion_result = (
        run_deletion_test(
            image=original_image,
            ranking=ranking,
            model=model_context.model,
            target_class=target_class,
            preprocess_fn=model_context.preprocess_fn,
            num_steps=num_steps
        )
    )

    insertion_result = (
        run_insertion_test(
            image=original_image,
            ranking=ranking,
            model=model_context.model,
            target_class=target_class,
            preprocess_fn=model_context.preprocess_fn,
            num_steps=num_steps
        )
    )

    return {
        "insertion_auc":insertion_result["auc"],
        "deletion_auc":deletion_result["auc"],
        "insertion_curve_x":insertion_result["curve"]["x"],
        "insertion_curve_y":insertion_result["curve"]["y"],
        "deletion_curve_x":deletion_result["curve"]["x"],
        "deletion_curve_y":deletion_result["curve"]["y"]
    }