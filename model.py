"""Custom CNN for 48x48 grayscale facial expression classification."""
from tensorflow import keras
from tensorflow.keras import layers

from config import EMOTIONS, IMG_SIZE


def conv_block(x, filters, dropout):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return layers.Dropout(dropout)(x)


def build_model(num_classes=len(EMOTIONS)):
    """Input: float pixels in [0, 255], shape (48, 48, 1).

    Rescaling and data augmentation live inside the model, so inference only
    needs raw 0-255 pixels. Augmentation layers are active only during training.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # augmentation (training only)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)

    x = conv_block(x, 32, 0.25)    # 48 -> 24
    x = conv_block(x, 64, 0.25)    # 24 -> 12
    x = conv_block(x, 128, 0.3)    # 12 -> 6
    x = conv_block(x, 256, 0.3)    # 6  -> 3

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="emotion_cnn")


if __name__ == "__main__":
    build_model().summary()
