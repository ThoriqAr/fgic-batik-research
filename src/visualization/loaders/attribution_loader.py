from pathlib import Path

import numpy as np


def load_attribution(npz_path):

    data = np.load(
        npz_path,
        allow_pickle=True
    )

    result = {}

    for key in data.files:

        value = data[key]

        if value.shape == ():

            result[key] = value.item()

        else:

            result[key] = value

    return result