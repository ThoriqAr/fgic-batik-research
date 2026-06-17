import matplotlib.pyplot as plt


def build_gradcam_figure(
    image,
    heatmap,
    overlay,
    attribution,
):

    ground_truth = attribution["ground_truth"]
    pred_label = attribution["pred_label"]
    confidence = attribution["confidence"]

    status = (

        "BENAR"

        if ground_truth == pred_label

        else "SALAH"
        
    )

    title_text = (

        f"Method : GradCAM\n"
        f"Ground Truth : {ground_truth}\n"
        f"Prediction : {pred_label}\n"
        f"Confidence : {confidence:.2%}\n"
        f"Status : {status}"

    )

    fig = plt.figure(
        figsize=(15, 7)
    )

    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(heatmap, cmap="inferno")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title_text, fontsize=12)

    plt.tight_layout()

    return fig