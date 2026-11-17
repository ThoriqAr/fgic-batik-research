import numpy as np

from tensorflow.keras.models import Model


def build_embedding_extractor(
    model,
    embedding_layer: str
):

    return Model(

        inputs=model.input,

        outputs=model.get_layer(
            embedding_layer
        ).output
    )


def generate_embedding(
    extractor,
    image_batch
):

    embedding = extractor.predict(
        image_batch,
        verbose=0
    )

    return embedding.astype(np.float32)