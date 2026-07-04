import numpy as np


def load_evaluation_target(
    sample
):

    return np.array(

        [
            sample.pred_index
        ],

        dtype=np.int64
    )