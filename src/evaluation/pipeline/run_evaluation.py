import time

from src.evaluation.loaders.experiment_loader import (
    load_experiment
)

from src.evaluation.pipeline.evaluation_pipeline import (
    evaluate_experiment
)

from src.evaluation.artifacts.evaluation_artifacts import (
    create_evaluation_artifacts
)

from src.evaluation.artifacts.summary import (
    build_summary,
    save_summary
)

from src.evaluation.artifacts.config import (
    build_evaluation_config,
    save_evaluation_config
)

from src.evaluation.artifacts.evaluation_result import (
    save_results_csv
)

from src.evaluation.artifacts.error_log import (
    save_error_log
)


def run_evaluation(
    evaluation_config,
    num_steps=20
):
  
    print(
        "\n[1/5] Load Experiment..."
    )

    experiment = (
        load_experiment(
            evaluation_config.attribution_dir
        )
    )

    print(
        "\n[2/5] Create Artifacts..."
    )

    artifacts = (
        create_evaluation_artifacts(
            evaluation_config.experiment_name
        )
    )

    print(
        "\n[3/5] Evaluate Samples..."
    )

    start_time = time.time()

    results, error_records = (
        evaluate_experiment(
            experiment=experiment,
            evaluation_config=evaluation_config,
            num_steps=num_steps
        )
    )

    runtime_seconds = (
        time.time() - start_time
    )

    print(
        "\n[4/5] Save Results..."
    )

    save_results_csv(
        results,
        artifacts.results_csv
    )

    save_error_log(
        error_records,
        artifacts.errors_csv
    )

    summary = (
        build_summary(
            results
        )
    )

    save_summary(
        summary,
        artifacts.summary_json
    )

    config = (
        build_evaluation_config(
            experiment=experiment,
            evaluation_config=evaluation_config,
            results=results,
            runtime_seconds=runtime_seconds,
            error_records=error_records
        )
    )

    save_evaluation_config(
        config,
        artifacts.config_json
    )

    print(
        "\n[5/5] Evaluation Completed"
    )

    print(
        f"Output : "
        f"{artifacts.experiment_dir}"
    )