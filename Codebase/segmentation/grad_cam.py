# Codebase/segmentation/grad_cam.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2


def load_and_preprocess_img(img_path, target_size, normalize=True):
    """Load and preprocess image."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if normalize:
        x = x / 255.0

    return img, x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Activation-based CAM (NO gradients).
    Very stable for Keras 3. Works for any CNN.
    """

    # Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Build model that outputs activations
    activation_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    # Forward pass
    conv_output = activation_model(img_array)[0]  # shape (H, W, C)

    # Average activations across channels
    heatmap = tf.reduce_mean(conv_output, axis=-1)

    # ReLU + normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4):
    """Overlay heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (orig_img.size[0], orig_img.size[1]))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = image.img_to_array(orig_img)
    superimposed = heatmap_color * alpha + img_array
    return np.clip(superimposed, 0, 255).astype("uint8")


def show_gradcam_triplet(orig_img, heatmap, overlay_img):
    """Show original, heatmap, and overlay."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(orig_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay_img.astype("uint8"))
    plt.axis("off")

    plt.tight_layout()
    plt.show()
