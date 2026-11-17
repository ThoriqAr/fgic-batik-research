import random


def build_stratified_sampling(
    samples,
    total_samples=10,
    seed=42
):

    random.seed(
        seed
    )

    grouped = {}

    for sample in samples:

        class_name = (
            sample.ground_truth
        )

        grouped.setdefault(

            class_name,

            []

        ).append(
            sample
        )

    classes = list(
        grouped.keys()
    )

    random.shuffle(
        classes
    )

    if total_samples > len(classes):

        raise ValueError(

            "total_samples "
            "cannot exceed "
            "number of classes."

        )

    selected = []

    selected_classes = classes[
        :total_samples
    ]

    for class_name in selected_classes:

        candidates = grouped[
            class_name
        ]

        selected.append(

            random.choice(
                candidates
            )

        )

    selected.sort(

        key=lambda sample:
        sample.ground_truth

    )

    return selected