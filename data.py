"""Dataset loading for FER-2013.

Supported layouts inside the data directory:
  A) fer2013.csv                       (Kaggle 'fer2013' CSV: emotion,pixels,Usage)
  B) train/<emotion>/*.jpg + test/<emotion>/*.jpg   (Kaggle 'msambare/fer2013')
"""
import os

import cv2
import numpy as np

from config import EMOTIONS, IMG_SIZE

# In fer2013.csv: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
_CSV_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 4}


def _load_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    pixels = np.stack(
        [np.array(p.split(), dtype=np.uint8) for p in df["pixels"].values]
    ).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array([_CSV_TO_IDX[int(e)] for e in df["emotion"].values])

    is_train = (df["Usage"] == "Training").values
    x = pixels.astype("float32")
    return x[is_train], labels[is_train], x[~is_train], labels[~is_train]


def _load_folder(root):
    xs, ys = [], []
    for idx, name in enumerate(EMOTIONS):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            xs.append(img)
            ys.append(idx)
    if not xs:
        raise FileNotFoundError(f"No images found under {root}")
    x = np.array(xs, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x, np.array(ys)


def load_dataset(data_dir):
    """Returns x_train, y_train, x_val, y_val (x in 0-255 float32, y int labels)."""
    csv_path = os.path.join(data_dir, "fer2013.csv")
    train_dir = os.path.join(data_dir, "train")

    if os.path.isfile(csv_path):
        print(f"Loading CSV dataset: {csv_path}")
        return _load_csv(csv_path)

    if os.path.isdir(train_dir):
        print(f"Loading folder dataset: {data_dir}")
        x_train, y_train = _load_folder(train_dir)
        val_dir = next(
            (os.path.join(data_dir, d) for d in ("test", "val", "validation")
             if os.path.isdir(os.path.join(data_dir, d))),
            None,
        )
        if val_dir:
            x_val, y_val = _load_folder(val_dir)
        else:  # hold out 10%
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(x_train))
            cut = int(0.9 * len(perm))
            x_val, y_val = x_train[perm[cut:]], y_train[perm[cut:]]
            x_train, y_train = x_train[perm[:cut]], y_train[perm[:cut]]
        return x_train, y_train, x_val, y_val

    raise FileNotFoundError(
        f"FER-2013 not found in '{data_dir}'.\n"
        "Put either fer2013.csv or the train/ and test/ folders there. See README.md."
    )
