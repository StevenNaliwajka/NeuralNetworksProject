# Codebase/segmentation/grad_cam.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2


def load_and_preprocess_img(img_path, target_size, normalize=True):
    """
    Load an image from disk and resize to target_size.
    Returns (orig_pil_image, preprocessed_np_array[1, H, W, 3])
    """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if normalize:
        x = x / 255.0

    return img, x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, class_index=None):
    """
    Compute Grad-CAM heatmap for a given model and input image.
    """

    # 0. Make sure the model has been called at least once
    #    (Keras 3 sometimes complains "sequential has never been called").
    try:
        _ = model.outputs
    except AttributeError:
        _ = model(img_array)

    # 1. Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # 2. Build a model that outputs (conv_features, predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]],
    )

    # 3. Forward pass + record gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    # 4. Gradients of class score w.r.t conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # 5. Global-average-pool over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Weight conv maps by pooled grads
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. ReLU and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4):
    """
    Overlay heatmap on top of a PIL image.

    Returns
    -------
    np.ndarray
        Superimposed RGB image.
    """
    # Convert to 0–255
    heatmap = np.uint8(255 * heatmap)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (orig_img.size[0], orig_img.size[1]))

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert original PIL image to array
    img_array = image.img_to_array(orig_img)

    # Superimpose heatmap onto original image
    superimposed = heatmap_color * alpha + img_array
    superimposed = np.clip(superimposed, 0, 255).astype("uint8")

    return superimposed


def show_gradcam_triplet(orig_img, heatmap, overlay_img):
    """
    Show original image, heatmap, and overlay next to each other.
    """
    plt.figure(figsize=(12, 4))

    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig_img)
    plt.axis("off")

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay_img.astype("uint8"))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
