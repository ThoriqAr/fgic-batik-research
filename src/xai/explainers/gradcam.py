import numpy as np
import tensorflow as tf


def build_gradcam_heatmap(
    processed_image,
    model,
    pred_index,
    backbone_name,
    last_conv_layer_name
):

    # ambil backbone model dari model utama
    backbone = model.get_layer(
        backbone_name
    )

    # ambil layer konvulusi terakhir dari backbone
    last_conv_layer = (
        backbone.get_layer(
            last_conv_layer_name
        )
    )

    # buat model baru yang hanya berisi backbone dan layer konvulusi terakhir
    # model ini akan digunakan untuk mengambil feature map
    last_conv_model = (
        tf.keras.models.Model(
            inputs=backbone.input,
            outputs=last_conv_layer.output
        )
    )

    # membangun kembali classifier head untuk menghubungkan feature map ke layer klasifikasi
    # Bertujuan untuk memperoleh skor/logit kelas target
    classifier_input = (
        tf.keras.Input(
            shape=last_conv_layer.output.shape[1:]
        )
    )

    x = classifier_input

    passed_backbone = False

    # Meneruskan feature map melalui layer setelah backbone hingga mencapai output klasifikasi
    for layer in model.layers:

        if layer.name == backbone_name:

            passed_backbone = True

            continue

        if passed_backbone:

            x = layer(x)

    classifier_model = (
        tf.keras.models.Model(
            classifier_input,
            x
        )
    )

    # Forward propagation untuk menghasilkan feature map dari citra
    # Lalu meneruskannya ke classifier untuk memperoleh output model
    with tf.GradientTape() as tape:

        # menghasilkan feature map dari last convolutional layer
        conv_output = (
            last_conv_model(
                processed_image
            )
        )

        # Feature map dipantau agar gradient terhadap feature map
        # dapat dihitung pada proses backward propagation.
        tape.watch(
            conv_output
        )

        # Meneruskan feature map ke classifier head
        predictions = (
            classifier_model(
                conv_output
            )
        )

        # Mengambil skor/logit untuk kelas target
        # berdasarkan pred_index.
        class_channel = (
            predictions[:, pred_index]
        )

    # Backward propagation, bertujuan untuk menghitung gradient dari skor kelas target
    # terhadap setiap nilai pada feature map.
    grads = tape.gradient(
        class_channel,
        conv_output
    )

    # Global Average Pooling, gradient di rata-ratakan pada seluruh posisi spasial
    # untuk memperoleh importance weight setiap feature map.
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Mengambil featuer map untuk satu sampel
    conv_output = conv_output[0]

    # Setiap feature map dikalikan dengan importance weightnya,
    # kemudian seluruh feature map dijumlahkan
    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    # ReLU, mempertahankan kontribusi posistif dan menghilangkan kontribusi negatif
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalisasi, heatmap dibagi menjadi nilai maksimumnya
    # sehingga nilai heatmap berada pada rentang sekitar 0-1
    heatmap /= (
        tf.reduce_max(
            heatmap
        ) + 1e-8
    )

    # mengubah tensor menjadi numpy array
    return heatmap.numpy()


def generate_gradcam(
    model,
    processed_image,
    pred_index,
    backbone_name,
    last_conv_layer_name,
):

    raw_heatmap = (
        build_gradcam_heatmap(
            processed_image=processed_image,
            model=model,
            pred_index=pred_index,
            backbone_name=backbone_name,
            last_conv_layer_name=last_conv_layer_name
        )
    )

    return {

        "raw_heatmap":
            raw_heatmap.astype(
                np.float32
            )

    }