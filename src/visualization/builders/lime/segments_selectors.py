def select_segments(
    segment_ids,
    weights,
    mode="positive",
    top_k=None
):

    pairs = list(
        zip(segment_ids, weights)
    )

    if mode == "positive":

        pairs = [

            (seg_id, weight)

            for seg_id, weight in pairs

            if weight > 0
        ]

        pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

    elif mode == "negative":

        pairs = [

            (seg_id, weight)

            for seg_id, weight in pairs

            if weight < 0
        ]

        pairs.sort(
            key=lambda x: x[1]
        )

    elif mode == "absolute":

        pairs.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

    elif mode == "signed":

        pairs.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

    else:

        raise ValueError(
            f"Unknown mode: {mode}"
        )

    if top_k is not None:

        pairs = pairs[:top_k]

    return pairs