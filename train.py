"""Train the CNN on FER-2013.   python train.py --epochs 40"""
import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from config import DATA_DIR, EMOTIONS, MODEL_PATH
from data import load_dataset
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="use at most N training images per emotion (fast mode); 0 = all")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(42)
    x_train, y_train, x_val, y_val = load_dataset(args.data_dir)
    if args.max_per_class > 0:  # fast mode: random subset, same for every run
        rng = np.random.default_rng(42)
        keep = np.concatenate([
            rng.permutation(np.where(y_train == c)[0])[:args.max_per_class]
            for c in range(len(EMOTIONS))])
        x_train, y_train = x_train[keep], y_train[keep]
        if len(x_val) > 2000:
            keep = rng.permutation(len(x_val))[:2000]
            x_val, y_val = x_val[keep], y_val[keep]
        print(f"FAST MODE: <= {args.max_per_class} images per emotion")
    print(f"Train: {x_train.shape}  Validation: {x_val.shape}")
    print("Train class counts:", dict(zip(EMOTIONS, np.bincount(y_train, minlength=len(EMOTIONS)))))

    # Soft class weighting to help rare classes (e.g. 'disgust')
    classes = np.arange(len(EMOTIONS))
    present = np.unique(y_train)
    w = compute_class_weight("balanced", classes=present, y=y_train)
    class_weight = {int(c): float(np.sqrt(v)) for c, v in zip(present, w)}

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                      restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Final evaluation with the best saved model
    best = keras.models.load_model(args.out)
    probs = best.predict(x_val, batch_size=256, verbose=0)
    pred = probs.argmax(1)
    labels = list(range(len(EMOTIONS)))
    print("\nBest model validation accuracy: %.4f" % (pred == y_val).mean())
    print(classification_report(y_val, pred, labels=labels,
                                target_names=EMOTIONS, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_val, pred, labels=labels))

    # Training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history.history["accuracy"], label="train")
        ax[0].plot(history.history["val_accuracy"], label="val")
        ax[0].set_title("Accuracy"); ax[0].legend()
        ax[1].plot(history.history["loss"], label="train")
        ax[1].plot(history.history["val_loss"], label="val")
        ax[1].set_title("Loss"); ax[1].legend()
        plot_path = os.path.splitext(args.out)[0] + "_history.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        print("Saved training curves to", plot_path)
    except Exception as e:  # plotting is optional
        print("Skipped plot:", e)

    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
