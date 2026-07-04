from tqdm.auto import tqdm

from src.visualization.loaders.attribution_loader import (
    load_attribution
)

from src.evaluation.loaders.model_loader import (
    load_model_context
)

from src.evaluation.loaders.sample_selector import (
    select_samples
)

from src.evaluation.loaders.evaluation_original_image_loader import (
    load_evaluation_original_image
)

from src.evaluation.methods.faithfulness import (
    evaluate_faithfulness
)

from src.evaluation.methods.stability import (
    evaluate_stability
)

from src.evaluation.artifacts.evaluation_result import (
    build_evaluation_result
)


def evaluate_experiment(
    experiment,
    evaluation_config,
    num_steps=20,
):
  
    model_context = (
        load_model_context(
            experiment
        )
    )

    samples = (
        experiment.manifest
        .get_samples()
    )

    samples = (
        select_samples(
            samples=samples,
            evaluation_config=evaluation_config
        )
    )

    results = []

    error_count = 0

    error_records = []

    pbar = tqdm(
        samples,
        desc="Evaluating Samples",
        unit="sample"
    )

    for sample in pbar:

        try:

            attribution = (
                load_attribution(
                    sample.attribution_path
                )
            )

            original_image = (
                load_evaluation_original_image(
                    sample,
                    model_context
                )
            )

            faithfulness_result = (
                evaluate_faithfulness(
                    sample=sample,
                    original_image=original_image,
                    model_context=model_context,
                    num_steps=num_steps
                )
            )

            stability_result = (
                evaluate_stability(
                    sample=sample,
                    original_image=original_image,
                    experiment=experiment,
                    model_context=model_context
                )
            )

            result = (
                build_evaluation_result(
                    sample=sample,
                    attribution=attribution,
                    faithfulness_result=faithfulness_result,
                    stability_result=stability_result
                )
            )

            results.append(
                result
            )

        except Exception as e:

            error_count += 1

            error_records.append(
                {
                    "attribution_id":
                        sample.attribution_id,

                    "error":
                        repr(e)
                }
            )

            print(
                f"ERROR [{error_count}] "
                f"{sample.attribution_id}"
            )

            print(repr(e))

            continue

    print(
        f"Completed : "
        f"{len(results)} samples"
    )

    print(
        f"Errors : "
        f"{error_count}"
    )

    return (
        results,
        error_records
    )