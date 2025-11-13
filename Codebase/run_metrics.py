from typing import Sequence, Mapping, Any


## Create Data Config class to pass reference between methods.
class RunMetrics:
    ## Data Storage
    datagen: float

    ## File Path
    input_path = Path("train.zip")
    output_path = Path("data/cats_vs_dogs")
    ## Qty of images to intake
    n_total = 10000
    ## Size every photo is re-sized too at first. New Models Override.
    target_size = (128, 128)
    ## Targeted IDs to remove noisy data. (mod4 slides).
    blacklist_ids = {5604, 6413, 8736, 8898, 9188, 9517, 10161, 10190, 10237, 10401, 10797, 11186}
    extract_dir = output_path / "_extracted_all"

    ## Training
    total_time: float
    epoch_times: Sequence[float]
    avg_epoch: float
    epochs_run: int
    steps_per_epoch: int

    ## Kerras
    hist: Mapping[str, Sequence[float]]
    acc: Sequence[float]
    val_acc: Sequence[float]
    loss: Sequence[float]
    val_loss: Sequence[float]
    best_idx: int
    gap: float

    ## Test Matrix.
    test_loss: float
    test_acc: float
    total_test_time: float
    num_test_images: int
    steps_test: int
    avg_time_per_step: float
    avg_time_per_image: float

    @property
    def best_val_acc(self) -> float:
        return float(self.val_acc[self.best_idx])

    @property
    def best_val_loss(self) -> float:
        return float(self.val_loss[self.best_idx])


m = RunMetrics()
print("Finished")