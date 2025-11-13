def inject_mobilenet_v2():
    ## A friend spoke highly of mobilenet V2.
    ## It seems to work better than VGG-16. Smaller and newer.
    ## Google says Mobilenet Came out 4 years later 2018.

    ## Restarts the model so old training sessions dont carry over.
    tf.keras.backend.clear_session()

    ## Pretrained MobileNetV2 base (no top, Imagenet weights)
    target_size = (160, 160)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*target_size, 3)
    )
    # freeze feature extractor
    base.trainable = False

    ## Custom classification head
    inputs = layers.Input(shape=(*target_size, 3))
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