"""Creates a tiny SYNTHETIC dataset in FER-2013 folder format.

ONLY for checking that the code runs end-to-end. It is not real data and a model
trained on it will NOT recognise real faces. Use the real FER-2013 for that.
"""
import os
import sys

import cv2
import numpy as np

from config import EMOTIONS, IMG_SIZE


def draw(idx, rng):
    img = np.full((IMG_SIZE, IMG_SIZE), 90, np.uint8)
    cv2.circle(img, (24, 24), 20, 170, -1)
    # eyes
    cv2.circle(img, (16, 19), 2 + idx % 3, 40, -1)
    cv2.circle(img, (32, 19), 2 + idx % 3, 40, -1)
    # brows slant and mouth curvature depend on the class
    tilt = (idx - 3) * 2
    cv2.line(img, (11, 12 + tilt // 2), (21, 12 - tilt // 2), 30, 2)
    cv2.line(img, (27, 12 - tilt // 2), (37, 12 + tilt // 2), 30, 2)
    cv2.ellipse(img, (24, 32), (8, 3 + 2 * abs(idx - 3)), 0, 0,
                180 if idx % 2 else -180, 30, 2)
    img = np.roll(img, rng.integers(-2, 3, 2), axis=(0, 1))
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main(out="data_sample", n_train=120, n_test=30):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("test", n_test)):
        for idx, name in enumerate(EMOTIONS):
            folder = os.path.join(out, split, name)
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                cv2.imwrite(os.path.join(folder, f"{i}.png"), draw(idx, rng))
    print("Synthetic sample data written to", out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["data_sample"]))
