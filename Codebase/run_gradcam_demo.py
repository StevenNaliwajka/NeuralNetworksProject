from pathlib import Path
import tensorflow as tf

from segmentation.grad_cam import (
    load_and_preprocess_img,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    show_gradcam_triplet,
)

def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    data_dir = project_root / "data"

    # Load model
    model_path = models_dir / "cats_vs_dogs_functional.keras"
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Find image
    example_img = next(data_dir.rglob("*.jpg"))
    print(f"Using image: {example_img}")

    # Preprocess
    target_size = model.input_shape[1:3]
    orig_img, x = load_and_preprocess_img(str(example_img), target_size)

    # Find last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    print(f"Last conv layer: {last_conv}")

    # Generate heatmap
    heatmap = make_gradcam_heatmap(x, model, last_conv)

    # Overlay
    overlay = overlay_heatmap_on_image(orig_img, heatmap)

    # Display results
    show_gradcam_triplet(orig_img, heatmap, overlay)


if __name__ == "__main__":
    main()
