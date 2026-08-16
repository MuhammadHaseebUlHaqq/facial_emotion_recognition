"""
Facial Emotion Recognition — Training Script

Usage:
    python train_model.py                          # uses default paths ../train and ../test
    python train_model.py --data_dir /path/to/fer  # custom base dir containing train/ and test/

The FER-2013 folder structure should be:
    <data_dir>/train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
    <data_dir>/test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from model import build_model, IMG_SIZE, NUM_CLASSES, EMOTIONS

# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50
VAL_SPLIT = 0.15

np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_images_from_folder(base_path: str, split: str):
    """Load images from base_path/split/emotion_name/ folders.

    Returns normalised arrays X (N, 48, 48, 1) and one-hot labels y.
    """
    emotions_lower = [e.lower() for e in EMOTIONS]
    X, y = [], []
    split_path = os.path.join(base_path, split)

    for emotion_idx, emotion in enumerate(emotions_lower):
        folder = os.path.join(split_path, emotion)
        if not os.path.exists(folder):
            print(f"  ⚠ Warning: missing folder → {folder}")
            continue

        count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0)
            y.append(emotion_idx)
            count += 1

        print(f"  {EMOTIONS[emotion_idx]:<10} {count} images")

    return np.array(X, dtype=np.float32), to_categorical(y, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the FER CNN model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, ".."),
        help="Base directory containing train/ and test/ folders (default: parent of script dir)",
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    # ------ Load data ------
    print("Loading training data...")
    X_train_full, y_train_full = load_images_from_folder(data_dir, "train")
    print(f"\nLoading test data...")
    X_test, y_test = load_images_from_folder(data_dir, "test")

    if len(X_train_full) == 0 or len(X_test) == 0:
        print("\n✗ No images found. Make sure the FER-2013 dataset is placed at:")
        print(f"   {data_dir}/train/<emotion>/ and {data_dir}/test/<emotion>/")
        sys.exit(1)

    # ------ Train / Val split ------
    n = len(X_train_full)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - VAL_SPLIT))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    print(f"\nTrain : {X_train.shape}")
    print(f"Val   : {X_val.shape}")
    print(f"Test  : {X_test.shape}")

    # ------ Class weights ------
    train_labels = y_train.argmax(axis=1)
    weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_labels)
    class_weight_dict = dict(enumerate(weights))
    print("\nClass weights:")
    for i, e in enumerate(EMOTIONS):
        print(f"  {e:<10} {weights[i]:.3f}")

    # ------ Data augmentation ------
    aug_gen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
    )

    # ------ Build model ------
    model = build_model()
    model.summary()

    # ------ Callbacks ------
    model_path = os.path.join(SCRIPT_DIR, "emotion_model.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    # ------ Train ------
    print("\n🚀 Training started …\n")
    history = model.fit(
        aug_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ------ Evaluate ------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ Test accuracy : {accuracy:.4f}")
    print(f"  Test loss     : {loss:.4f}")
    print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
