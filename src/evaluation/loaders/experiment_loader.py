import json
import pandas as pd
from pathlib import Path

from src.evaluation.artifacts.manifest import (
    EvaluationManifest
)

from src.evaluation.artifacts.evaluation_experiment import (
    EvaluationExperiment
)


def load_experiment_config(
    config_path
):

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)
    

def load_manifest(
    manifest_path
):

    df = pd.read_csv(
        manifest_path
    )

    return EvaluationManifest(
        dataframe=df
    )


def load_experiment(
    attribution_dir
):

    attribution_dir = Path(
        attribution_dir
    )

    config = (
        load_experiment_config(
            attribution_dir /
            "config.json"
        )
    )

    manifest = (
        load_manifest(
            attribution_dir /
            "manifest.csv"
        )
    )

    return EvaluationExperiment(
        experiment_name=config["experiment_name"],
        model_name=config["model_name"],
        xai_method=config["xai_method"],
        attribution_dir=attribution_dir,
        manifest=manifest,
        config=config
    )