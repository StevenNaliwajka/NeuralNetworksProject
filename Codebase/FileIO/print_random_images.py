import random

from PIL import Image

from Codebase import run_metrics
from Codebase.run_metrics import RunMetrics

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def print_random_images(m: RunMetrics):
    ## Input
    input_path = m.input_path
    ## Output
    output_path = m.output_path
    ## Pick Random Image from Training folder 60% split
    cat_path = random.choice(list((output_path / "train" / "cats").glob("*.jpg")))
    dog_path = random.choice(list((output_path / "train" / "dogs").glob("*.jpg")))

    ## Open.
    cat_img = Image.open(cat_path)
    dog_img = Image.open(dog_path)

    ## PLOT together.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cat_img)
    axes[0].set_title("Cat")
    axes[1].imshow(dog_img)
    axes[1].set_title("Dog")

    ## Keeping Axis-es allowes me to verify I have standarddiz-ed the sizees.
    plt.show()

    ## Print IDs
    print("Cat image:", cat_path.name)
    print("Dog image:", dog_path.name)