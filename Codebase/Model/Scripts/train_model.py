def train_model(m: RunMetrics):
    ## Where data is saved.
    output_path = Path(r"data/cats_vs_dogs")
    ## Size of Each EPOCH
    batch_size = 32
    ## MAX EPOCH.
    epochs = 10
    target_size = (128, 128)

    ## Create Data loader object
    m.datagen = ImageDataGenerator()
    ## Setting settings for training set.
    ## Output path,
    ## Batch size,
    ## Class_mode - Set to binary since we are detecting between cats/dogs.
    ## No need to bother with shuffle/resize since thats handled in fileIO above.
    ## Strange that it defaults to 256x256. I HAVE to specify 128,128
    train_gen = m.datagen.flow_from_directory(
        output_path / "train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    ## Same for validation set
    val_gen = m.datagen.flow_from_directory(
        output_path / "val",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    ## The "training control mechanism"
    ## Stops early. After 2 'failed epochs' w/o any benefit.
    ## Reloads the 'best' weights.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor="val_loss"
        )
    ]

    ## Create my own callback so I can have proper time documentation/
    class TimeHistory(tf.keras.callbacks.Callback):
        epoch_times = None
        total_time = None
        avg_epoch = None
        epochs_run = None
        steps_per_epoch = None

        def on_train_begin(self, logs=None):
            self.epoch_times = []

        def on_epoch_begin(self, epoch, logs=None):
            ## Document 'start epoch time'.
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            ## Calc + Document the epoch end.
            self.epoch_times.append(time.time() - self._t0)

    ## instance of time
    time_cb = TimeHistory()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        ## Pass in time callback
        callbacks=[*callbacks, time_cb],
        verbose=1
    )

    ## Compute Times.
    epoch_times = np.array(time_cb.epoch_times, dtype=float)
    total_time = epoch_times.sum()
    avg_epoch = epoch_times.mean()
    epochs_run = len(epoch_times)
    steps_per_epoch = int(np.ceil(train_gen.samples / batch_size))

    ## Summary
    print(f"Steps/epoch: {steps_per_epoch}")
    print(f"Epochs run: {epochs_run}/{epochs}")
    print(f"Avg epoch time: {avg_epoch:.2f} s")
    print(f"Total training time: {total_time:.2f} s")

    hist = history.history
    acc, val_acc = np.array(hist["accuracy"]), np.array(hist["val_accuracy"])
    loss, val_loss = np.array(hist["loss"]), np.array(hist["val_loss"])
    best_idx = int(val_acc.argmax())

    gap = float(val_acc[best_idx] - acc[best_idx])
    print(f"Generalization gap: {gap:+.4f}")

    m.total_time = total_time
    m.epoch_times = epoch_times
    m.avg_epoch = avg_epoch
    m.epochs_run = int(epochs_run)
    m.steps_per_epoch = steps_per_epoch
    m.hist = hist
    m.acc = acc
    m.val_acc = val_acc
    m.loss = loss
    m.val_loss = val_loss
    m.best_idx = best_idx
    m.gap = gap


train_model(m)
print("Finished")