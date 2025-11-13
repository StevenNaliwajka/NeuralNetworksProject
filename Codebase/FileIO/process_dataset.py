import random
import re
import shutil
import zipfile

from PIL import Image

from Codebase.run_metrics import RunMetrics


def process_dataset(m: RunMetrics):
    ## Input
    input = m.input_path
    ## Output
    output = m.output_path
    ## Qty of images to intake
    n_total = m.n_total
    ## Size every photo is re-sized too.
    target_size = m.target_size
    ## Targeted IDs to remove noisy data. (mod4 slides)
    blacklist_ids = m.blacklist_ids

    ## Extract all images into new dir.
    ## IF exist. Remove and replace.
    extract_dir = m.extract_dir
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    ## Unzips training data.
    with zipfile.ZipFile(input, "r") as zf:
        zf.extractall(extract_dir)

    print("Extracted to:", extract_dir)

    ## Display CAT or DOG Kaggle naming qtys with pattern 'cat.XXXXX.jpg', 'dog.XXXXX.jpg')
    all_imgs = list(extract_dir.rglob("*.jpg"))
    cats = [p for p in all_imgs if "cat" in p.name.lower()]
    dogs = [p for p in all_imgs if "dog" in p.name.lower()]
    print(f"Found cats: {len(cats)}, dogs: {len(dogs)}, total: {len(all_imgs)}")

    ## Filename Parser: Used for removing blacklisted data.
    id_re = re.compile(r'^(cat|dog)\.(\d+)\.(jpg|jpeg)$', re.IGNORECASE)

    ## Prunes pictures for any blacklisted IDs from mod4 slides.
    ## Removes both cat/dog version since I dont know which group it is.
    def keep_file(p):
        m = id_re.match(p.name)
        if not m:
            return True
        num = int(m.group(2))
        return num not in blacklist_ids

    cats_before, dogs_before = len(cats), len(dogs)
    cats = [p for p in cats if keep_file(p)]
    dogs = [p for p in dogs if keep_file(p)]

    ## Randomly re-numbers the pictures to ensure truely unique runs.
    num_each = min(len(cats), len(dogs), n_total // 2)
    random.shuffle(cats)
    random.shuffle(dogs)
    cats = cats[:num_each]
    dogs = dogs[:num_each]

    ## Run the math to split the Dataset... 60% Train, 15% Validation, 25% Testing as per reqs.

    n_train = int(round(n_total * 0.60))
    n_val = int(round(n_total * 0.15))
    n_test = n_total - n_train - n_val
    print(f"Per group -> train:{n_train}, val:{n_val}, test:{n_test}")

    ## Create Folders.
    for split in ["train", "val", "test"]:
        for cls in ["cats", "dogs"]:
            (output / split / cls).mkdir(parents=True, exist_ok=True)

    ## Populate Folders.
    ## Helper Method to bulk copy + resize.
    ## Applies some pre-processing tech from slides to support learning.
    ## - Converts to RGB Grid
    ## - Resize images to 128x128
    ## MAY NOT BE NEEDED IN RETROSPECT. I BELIVE EACH MODEL INHERENTLY RESIZES Images if its 'specified' version~
    ## IS not the right size. Saves compute time when handling bulk 128X128 imgs technichally.
    def resize_and_save(src_files, start_idx, n, dest_folder):
        dest_folder.mkdir(parents=True, exist_ok=True)
        for p in src_files[start_idx:start_idx + n]:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize(target_size)
                img.save(dest_folder / p.name)
            except Exception as e:
                print(f"Skipped {p.name}: {e}")
        return start_idx + n

    index = 0
    index = resize_and_save(cats, index, n_train, output / "train" / "cats")
    index = resize_and_save(cats, index, n_val, output / "val" / "cats")
    index = resize_and_save(cats, index, n_test, output / "test" / "cats")
    index = 0
    index = resize_and_save(dogs, index, n_train, output / "train" / "dogs")
    index = resize_and_save(dogs, index, n_val, output / "val" / "dogs")
    index = resize_and_save(dogs, index, n_test, output / "test" / "dogs")
    print(f"Dataset Populated Succuessfully.")

