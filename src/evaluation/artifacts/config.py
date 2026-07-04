import json
from datetime import datetime

def build_evaluation_config(
    experiment,
    evaluation_config,
    results,
    runtime_seconds,
    error_records
):
    
    successful_samples = len(results)

    samples_per_second = (
        successful_samples / runtime_seconds
        if runtime_seconds > 0
        else 0.0
    )

    samples_per_minute = (
        successful_samples / (runtime_seconds / 60)
        if runtime_seconds > 0
        else 0.0
    )

    successful_samples = len(results)

    failed_samples = len(
        error_records
    )

    total_processed = (
        successful_samples +
        failed_samples
    )
    
    return {

        "evaluation_experiment_name":
            evaluation_config.experiment_name,

        "attribution_experiment_name":
            experiment.experiment_name,

        "model_name":
            experiment.model_name,

        "xai_method":
            experiment.xai_method,

        "attribution_dir":
            experiment.attribution_dir.as_posix(),

        "metrics":
            evaluation_config.metrics,

        "perturbations":
            evaluation_config.perturbations,

        "total_samples":
            experiment.manifest.total_samples,

        "evaluated_samples":
            successful_samples,

        "failed_samples":
            failed_samples,

        "success_rate":
            (
                successful_samples /
                total_processed
            )
            if total_processed > 0
            else 0.0,

        "seed":
            evaluation_config.seed,

        "batch_size":
            evaluation_config.batch_size,

        "runtime_seconds":
            float(runtime_seconds),

        "runtime_minutes":
            float(runtime_seconds / 60),

        "samples_per_second":
            float(samples_per_second),

        "samples_per_minute":
            float(samples_per_minute),

        "created_at":
            datetime.now().isoformat()
    }


def save_evaluation_config(
    config,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )