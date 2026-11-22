"""Shared settings for the whole project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Class order is alphabetical (matches FER-2013 folder names).
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.environ.get(
    "EMOTION_MODEL", os.path.join(BASE_DIR, "models", "emotion_cnn.keras"))

# BGR colours used when drawing boxes / labels
COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 140, 0),
    "fear": (160, 32, 160),
    "happy": (0, 200, 255),
    "neutral": (200, 200, 200),
    "sad": (255, 120, 0),
    "surprise": (0, 165, 255),
}
