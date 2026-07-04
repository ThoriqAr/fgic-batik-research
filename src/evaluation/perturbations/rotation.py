import cv2
import numpy as np

def apply_rotation(
    image,
    angle=5
):

    height, width = (
        image.shape[:2]
    )

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(

        center,

        angle,

        1.0
    )

    rotated = cv2.warpAffine(

        image,

        matrix,

        (width, height),

        flags=cv2.INTER_LINEAR,

        borderMode=
            cv2.BORDER_REFLECT
    )

    return rotated.astype(
        np.uint8
    )