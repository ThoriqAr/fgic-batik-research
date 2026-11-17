from src.qualitative.loaders.load_manifests import (
    load_manifests
)

from src.qualitative.builders.build_intersection import (
    build_intersection
)

from src.qualitative.builders.build_stratified_sampling import (
    build_stratified_sampling
)

from src.qualitative.builders.build_qualitative_figure import (
    build_qualitative_figure
)

from src.qualitative.exporters.pdf_exporter import (
    export_pdf
)


def run_qualitative(
    mobilenet_gradcam_dir,
    mobilenet_lime_dir,
    resnet_gradcam_dir,
    resnet_lime_dir,
    total_samples=10,
    seed=42,
    output_path=None,
    gradcam_kwargs=None,
    lime_kwargs=None
):

    print(
        "\n[1/5] Load Manifests..."
    )

    manifests = load_manifests(
        mobilenet_gradcam_dir,
        mobilenet_lime_dir,
        resnet_gradcam_dir,
        resnet_lime_dir
    )

    print(
        "\n[2/5] Build Intersection..."
    )

    samples = build_intersection(
        manifests
    )

    print(
        "\n[3/5] Stratified Sampling..."
    )

    selected_samples = (
        build_stratified_sampling(
            samples=samples,
            total_samples=total_samples,
            seed=seed
        )
    )

    print(
        "\n[4/5] Build Qualitative Figures..."
    )

    visualizations = []

    for sample in selected_samples:

        visualization = (
            build_qualitative_figure(
                sample=sample,
                gradcam_kwargs=
                    gradcam_kwargs or {},
                lime_kwargs=
                    lime_kwargs or {}
            )
        )

        visualizations.append(
            visualization
        )

    print(
        "\n[5/5] Export PDF..."
    )

    export_pdf(
        visualizations=visualizations,
        output_path=output_path
    )

    print(
        "\nQualitative Evaluation Completed."
    )

    return visualizations