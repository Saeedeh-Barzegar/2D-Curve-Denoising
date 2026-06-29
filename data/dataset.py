import os
import numpy as np
import tensorflow as tf


CATEGORIES = ["4_points", "5_points", "6_points","7_points", "8_points", "9_points", "10_points"]


def load_curves(directory):

    curves = []
    for category in CATEGORIES:
        path = os.path.join(directory, category)
        if not os.path.isdir(path):
            print(f"[WARNING] Category folder not found: {path}")
            continue
        for filename in os.listdir(path):
            if filename.endswith(".npy"):
                curve = np.load(os.path.join(path, filename))
                curves.append(curve)
    return curves


def prepare_dataset(noisy_dir, clean_dir):

    noisy_curves = load_curves(noisy_dir)
    clean_curves = load_curves(clean_dir)

    assert len(noisy_curves) == len(clean_curves), (
        f"Mismatch: {len(noisy_curves)} noisy vs {len(clean_curves)} clean curves."
    )

    X = tf.keras.preprocessing.sequence.pad_sequences(noisy_curves, padding="post", dtype="float32")
    y = tf.keras.preprocessing.sequence.pad_sequences(clean_curves, padding="post", dtype="float32")

    return np.array(X), np.array(y)
