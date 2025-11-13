def inject_vgg_16():
    target_size = (128, 128)
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(*target_size, 3)
    )
    base.trainable = False

    inputs = layers.Input(shape=(*target_size, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="vgg16_transfer")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model = inject_vgg_16()
print("Finished")