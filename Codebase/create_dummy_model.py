from pathlib import Path
import tensorflow as tf

from segmentation.grad_cam import (
    load_and_preprocess_img,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    show_gradcam_triplet,
)


def main():
    # 1. Locate project root, models, and data folders
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    data_dir = project_root / "data"

    model_path = models_dir / "cats_vs_dogs_functional.keras"

    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print("➡️ Put a .keras model inside a folder named 'models' at project root.")
        return

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 🔴 IMPORTANT: build the model by calling it once on dummy data
    # This avoids the "sequential has never been called" error.
    input_shape = model.input_shape[1:]  # (H, W, C)
    dummy = tf.zeros((1, *input_shape))
    _ = model(dummy)

    # 2. Find an example image
    example_img = next(data_dir.rglob("*.jpg"), None)
    if example_img is None:
        print("⚠ No .jpg image found in /data/")
        print("➡️ Put any .jpg image inside a folder named 'data' at project root.")
        return

    print(f"Using example image: {example_img}")

    # 3. Preprocess for the model
    target_size = model.input_shape[1:3]
    orig_img, x = load_and_preprocess_img(str(example_img), target_size)

    # 4. Find the last Conv2D layer automatically
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break

    if last_conv_name is None:
        print("⚠ No Conv2D layer found in this model!")
        return

    print(f"Using last conv layer: {last_conv_name}")

    # 5. Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=last_conv_name)

    # 6. Overlay heatmap on original image
    overlay_img = overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4)

    # 7. Show results
    show_gradcam_triplet(orig_img, heatmap, overlay_img)


if __name__ == "__main__":
    main()
