"""Face detection (OpenCV Haar cascade) + emotion inference (trained CNN)."""
import os
import platform
import urllib.request

import cv2
import numpy as np
from tensorflow import keras

from config import BASE_DIR, COLORS, EMOTIONS, IMG_SIZE, MODEL_PATH

CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
               "data/haarcascades/" + CASCADE_NAME)


def find_cascade():
    """Locate the Haar cascade: bundled with OpenCV 4.x, else a local copy
    (downloaded once if missing - some OpenCV 5 wheels don't ship it)."""
    bundled = os.path.join(getattr(cv2, "data", None) and cv2.data.haarcascades or "",
                           CASCADE_NAME)
    if os.path.isfile(bundled):
        return bundled
    local = os.path.join(BASE_DIR, "models", CASCADE_NAME)
    if not os.path.isfile(local):
        print("Downloading face detector file ...")
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            urllib.request.urlretrieve(CASCADE_URL, local)
        except Exception as e:
            raise RuntimeError(
                f"Face detector file missing and download failed ({e}).\n"
                f"Download {CASCADE_URL}\nand save it as {local}") from e
    return local


def open_camera(index=0):
    """Open a webcam; returns a cv2.VideoCapture (check .isOpened())."""
    if platform.system() == "Windows":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


class EmotionRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found: {model_path}\n"
                "Train it first with:  python train.py"
            )
        self.model = keras.models.load_model(model_path)

        cascade_path = find_cascade()
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load face cascade: {cascade_path}")

    def detect_faces(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        return gray, [tuple(int(v) for v in f) for f in faces]

    def analyze(self, bgr):
        """Returns a list of dicts: box=(x,y,w,h), emotion, confidence, probs."""
        gray, boxes = self.detect_faces(bgr)
        if not boxes:
            return []
        batch = []
        for (x, y, w, h) in boxes:
            face = cv2.resize(gray[y:y + h, x:x + w], (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_AREA)
            batch.append(face)
        batch = np.asarray(batch, dtype="float32").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        probs = self.model(batch, training=False).numpy()

        results = []
        for box, p in zip(boxes, probs):
            i = int(p.argmax())
            results.append({
                "box": box,
                "emotion": EMOTIONS[i],
                "confidence": float(p[i]),
                "probs": {e: float(v) for e, v in zip(EMOTIONS, p)},
            })
        return results


def draw_results(bgr, results):
    """Draw boxes and labels on a copy of the frame."""
    out = bgr.copy()
    for r in results:
        x, y, w, h = r["box"]
        color = COLORS[r["emotion"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{r['emotion']} {r['confidence'] * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(y - th - base - 6, 0)
        cv2.rectangle(out, (x, top), (x + tw + 8, top + th + base + 6), color, -1)
        cv2.putText(out, label, (x + 4, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out
