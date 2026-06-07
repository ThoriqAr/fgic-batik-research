import hashlib
from pathlib import Path

def calculate_md5(file_path: Path) -> str:

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(4096),
            b""
        ):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def calculate_dataset_hash(
    train_csv,
    val_csv,
    test_csv
):

    hasher = hashlib.sha256()

    paths = [
        train_csv,
        val_csv,
        test_csv
    ]

    for path in paths:

        with open(path, "rb") as f:

            hasher.update(
                f.read()
            )

    return hasher.hexdigest()
    

def calculate_file_hash(path):

    with open(path, "rb") as f:

        return hashlib.sha256(
            f.read()
        ).hexdigest()
    

def format_learning_rate(lr):

    return (
        f"{lr:.0e}"
        .replace("e-0", "e-")
        .replace("e+0", "e+")
    )