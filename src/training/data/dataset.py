from src.data.loaders.load_dataset import (
    load_dataset
)

from src.preprocessing.augmentations.build_augmentation import (
    build_augmentation_layer
)


def create_training_datasets(
    train_config,
    preprocessing_fn
):
    
    augmentation_layer = (
    build_augmentation_layer(
            train_config.augmentation_config
        )
    )

    train_ds = load_dataset(
        csv_path=train_config.train_csv,
        preprocessing_config=train_config.preprocessing_config,
        augmentation_layer=augmentation_layer,
        is_training=True,
        batch_size=train_config.batch_size,
        shuffle=True,
        seed=train_config.seed,
        num_classes=train_config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    val_ds = load_dataset(
        csv_path=train_config.val_csv,
        preprocessing_config=train_config.preprocessing_config,
        augmentation_layer=None,
        is_training=False,
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        num_classes=train_config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    return train_ds, val_ds

def create_evaluation_dataset(
    train_config,
    preprocessing_fn
):
    test_ds = load_dataset(
        csv_path=train_config.test_csv,
        preprocessing_config=train_config.preprocessing_config,
        augmentation_layer=None,
        is_training=False,
        batch_size=train_config.batch_size,
        shuffle=False,
        seed=train_config.seed,
        num_classes=train_config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    return test_ds