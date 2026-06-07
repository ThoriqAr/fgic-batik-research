from tensorflow.keras.models import load_model

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from src.data.loaders.load_class_mapping import (
    get_class_names
)

from src.training.data.dataset import (
    create_datasets
)

from src.training.evaluation.metrics import (
    evaluate_model
)

from src.training.artifacts.evaluation_artifacts import (
    save_report,
    save_confusion_matrix,
    save_confusion_matrix_plot
)


def run_evaluation(
    model_path,
    config,
    output_dir
):
    
    evaluation_dir = (
        output_dir /
        f"{config.experiment_name}_evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    class_names = get_class_names()

    if config.model_name == "mobilenetv2":

        preprocess_input = mobilenet_preprocess

    elif config.model_name == "resnet50":

        preprocess_input = resnet_preprocess

    else:

        raise ValueError(
            config.model_name
        )

    _, _, test_ds = create_datasets(
        config,
        preprocess_input
    )

    print("\n[1/3] Load model...")

    model = load_model(
        model_path
    )

    print(f"Loaded model: {model_path}")

    print("\n[2/3] Evaluate model...")

    report, cm = evaluate_model(
        model,
        test_ds,
        class_names
    )

    print("\n[3/3] Save evaluation artifacts...")

    save_report(
        report,
        evaluation_dir /
        "classification_report.json"
    )

    save_confusion_matrix(
        cm,
        evaluation_dir /
        "confusion_matrix.csv"
    )

    save_confusion_matrix_plot(
        cm,
        class_names,
        evaluation_dir /
        "confusion_matrix.png"
    )

    print("Evaluation artifacts saved.")

    print(
        f"Artifacts : {evaluation_dir}"
    )