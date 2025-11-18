import tensorflow as tf
from tensorflow.keras import layers, models


def build_mobilenet_v2_model(
    input_shape=(224, 224, 3),
    num_classes=2,
    base_trainable=False,
    weights=None,
):
    """
    Build a MobileNetV2-based classifier (same idea as HW5).

    Parameters
    ----------
    input_shape : tuple
        Input image shape, default (224, 224, 3)
    num_classes : int
        Number of output classes (2 for binary classification)
    base_trainable : bool
        Whether to unfreeze the MobileNetV2 base for fine-tuning
    weights : str or None
        - None       -> random initialization (no download, safe on your Mac)
        - "imagenet" -> pretrained ImageNet weights (requires internet)

    Returns
    -------
    model : tf.keras.Model
        Compiled MobileNetV2 classifier
    """

    # 1. Load MobileNetV2 backbone (without top classifier)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,  # IMPORTANT: default is None to avoid download issues
    )

    # Freeze or unfreeze base model
    base_model.trainable = base_trainable

    # 2. Build classification head on top
    inputs = layers.Input(shape=input_shape)

    # Preprocessing that MobileNetV2 expects
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Feature extractor
    x = base_model(x, training=False)

    # Global pooling over spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Optional dropout for regularization
    x = layers.Dropout(0.2)(x)

    # Final classification layer(s)
    if num_classes == 1:
        # Binary classification using single sigmoid neuron
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        # Multi-class classification with softmax
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenet_v2_classifier")

    # 3. Compile model (similar style as HW5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics,
    )

    return model


if __name__ == "__main__":
    # Simple local test: build the model and print the summary.
    # Using weights=None so it does NOT try to download anything on your Mac.
    model = build_mobilenet_v2_model(
        input_shape=(224, 224, 3),
        num_classes=2,
        base_trainable=False,
        weights=None,  # change to "imagenet" on a machine with working SSL
    )
    model.summary()

