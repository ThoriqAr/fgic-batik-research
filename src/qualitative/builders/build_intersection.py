from src.qualitative.models.qualitative_sample import (
    QualitativeSample
)


def build_intersection(
    manifests
):

    grouped = {}

    for (
        experiment_name,
        manifest
    ) in manifests.items():

        for _, row in manifest.iterrows():

            sample_id = row["file_name"]

            if sample_id not in grouped:

                grouped[
                    sample_id
                ] = {
                    "ground_truth":row["ground_truth"],
                    "file_name":row["file_name"]
                }

            grouped[
                sample_id
            ][
                experiment_name
            ] = row[
                "attribution_path"
            ]

    samples = []

    required = [
        "mobilenet_gradcam",
        "mobilenet_lime",
        "resnet_gradcam",
        "resnet_lime"
    ]

    for sample in grouped.values():

        if not all(

            key in sample

            for key in required

        ):

            continue

        samples.append(

            QualitativeSample(

                sample_id=sample[
                    "file_name"
                ],

                ground_truth=sample[
                    "ground_truth"
                ],

                file_name=sample[
                    "file_name"
                ],

                mobilenet_gradcam=sample[
                    "mobilenet_gradcam"
                ],

                mobilenet_lime=sample[
                    "mobilenet_lime"
                ],

                resnet_gradcam=sample[
                    "resnet_gradcam"
                ],

                resnet_lime=sample[
                    "resnet_lime"
                ]

            )

        )

    return samples