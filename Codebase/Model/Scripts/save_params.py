def save_params(model_label: str | None = None):
    ## Define DIR if not already.
    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pick a label for this model
    if model_label is None:
        # use model.name if set, else default
        model_label = getattr(model, "name", "cnn")

    # Make it filename-safe
    safe_label = str(model_label).lower().replace(" ", "_")

    ## provide Paths that are UNIQUE per model type
    full_model_path = save_dir / f"cats_vs_dogs_{safe_label}.keras"
    weights_path = save_dir / f"cats_vs_dogs_{safe_label}.weights.h5"

    ## Save model.
    model.save(full_model_path)

    ## Save weights only.
    model.save_weights(weights_path)

    ## print
    print(f"\nSaved full model to: {full_model_path}")
    print(f"Saved weights only to: {weights_path}")


save_params(model.name)
print("Finished")