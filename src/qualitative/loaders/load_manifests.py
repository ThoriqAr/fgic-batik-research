import pandas as pd


def load_manifest(
    experiment_dir
):

    manifest_path = (
        experiment_dir /
        "manifest.csv"
    )

    manifest = (
        pd.read_csv(
            manifest_path
        )
    )

    return manifest


def load_manifests(
    mobilenet_gradcam_dir,
    mobilenet_lime_dir,
    resnet_gradcam_dir,
    resnet_lime_dir
):

    return {

        "mobilenet_gradcam":
            load_manifest(
                mobilenet_gradcam_dir
            ),

        "mobilenet_lime":
            load_manifest(
                mobilenet_lime_dir
            ),

        "resnet_gradcam":
            load_manifest(
                resnet_gradcam_dir
            ),

        "resnet_lime":
            load_manifest(
                resnet_lime_dir
            )

    }