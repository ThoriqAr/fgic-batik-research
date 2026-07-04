from skimage.metrics import (
    structural_similarity
)

from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.evaluation.standardization.attribution_to_heatmap import (
    attribution_to_heatmap
)

from src.evaluation.loaders.perturbed_attribution_loader import (
    load_perturbed_attribution
)

from src.evaluation.perturbations.registry import (
    PERTURBATION_REGISTRY
)


def compute_ssim(
    original_heatmap,
    perturbed_heatmap
):
    
    score = structural_similarity(
        original_heatmap,
        perturbed_heatmap,
        data_range=1.0
    )

    return float(score)


def evaluate_single_perturbation(
    sample,
    original_image,
    perturbation_fn,
    experiment,
    model_context
):
    
    attribution = (
        load_attribution(
            sample.attribution_path
        )
    )

    original_heatmap = (
        attribution_to_heatmap(
            attribution
        )
    )

    perturbed_image = (
        perturbation_fn(
            original_image
        )
    )

    perturbed_heatmap = (
        load_perturbed_attribution(
            sample=sample,
            perturbed_image=perturbed_image,
            experiment=experiment,
            model_context=model_context
        )
    )

    return compute_ssim(
        original_heatmap,
        perturbed_heatmap
    )


def evaluate_stability(
    sample,
    original_image,
    experiment,
    model_context
):
    
    scores = {}

    for (
        perturbation_name,
        perturbation_fn
    ) in (
        PERTURBATION_REGISTRY.items()
    ):

        score = (
            evaluate_single_perturbation(
                sample=sample,
                original_image=original_image,
                perturbation_fn=perturbation_fn,
                experiment=experiment,
                model_context=model_context
            )
        )

        scores[
            perturbation_name
        ] = score

    return scores