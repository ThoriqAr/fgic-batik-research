from src.data.loaders.load_dataset import load_dataset


def create_datasets(
    config,
    preprocessing_fn
):

    train_ds = load_dataset(
        csv_path=config.train_csv,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_classes=config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    val_ds = load_dataset(
        csv_path=config.val_csv,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_classes=config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    test_ds = load_dataset(
        csv_path=config.test_csv,
        image_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_classes=config.num_classes,
        preprocessing_fn=preprocessing_fn
    )

    return train_ds, val_ds, test_ds