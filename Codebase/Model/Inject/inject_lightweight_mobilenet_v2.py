## Scalled down test LightWeight MoblenetV2

def inject_lightweight_mobilenet_v2():
    ## ASKED CHATGPT FOR its pick of Neural Network to 'scale down' from mobilenetV2
    ## Provided me just that. A pre-trained mobilenet base.

    ## Restarts the model so old training sessions don't carry over.
    tf.keras.backend.clear_session()

    ## Lightweight MobileNetV2 base (no top, Imagenet weights, width multiplier 0.35)
    target_size = (160, 160)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        # width multiplier → scales channel depth
        alpha=0.35,
        input_shape=(*target_size, 3)
    )
    # freeze pretrained feature extractor
    base.trainable = False

    ## Custom classification head
    inputs = layers.Input(shape=(160, 160, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    ## Compile with same settings as before
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


print("Finished")