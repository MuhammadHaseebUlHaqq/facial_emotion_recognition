"""
Facial Emotion Recognition - CNN + BN + Dropout Model Architecture

This module defines the CNN model used for facial emotion recognition.
The architecture follows a "CNN + BatchNormalization + Dropout" pattern
designed for the FER-2013 dataset (48x48 grayscale images, 7 emotion classes).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def build_model() -> tf.keras.Model:
    """Build and compile the CNN + BN + Dropout emotion recognition model.

    Architecture
    ------------
    Block 1: Conv64 → BN → Conv64 → BN → MaxPool → Dropout(0.25)
    Block 2: Conv128 → BN → Conv128 → BN → MaxPool → Dropout(0.25)
    Block 3: Conv256 → BN → Conv256 → BN → MaxPool → Dropout(0.25)
    Dense  : Flatten → Dense512 → BN → Drop(0.5) → Dense256 → BN → Drop(0.5) → Dense7(softmax)

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    model = models.Sequential(name="FER_CNN_BN_Dropout")

    # ---- Block 1 ----
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                            input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Block 2 ----
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Block 3 ----
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Dense head ----
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    # ---- Compile ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


if __name__ == '__main__':
    m = build_model()
    m.summary()
