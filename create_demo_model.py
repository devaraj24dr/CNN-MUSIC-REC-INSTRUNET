"""
create_demo_model.py
====================
Builds and saves the InstruNet CNN model architecture as 'models/instrunet_cnn.h5'
using TensorFlow/Keras — matching exactly what streamlit_app.py expects.

Run this once locally:
    python create_demo_model.py

Then push the generated model:
    git add models/instrunet_cnn.h5
    git commit -m "feat: add CNN model"
    git push origin main
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================
# CONFIG — must match streamlit_app.py exactly
# =============================================
IMG_SIZE   = (128, 128)
N_CLASSES  = 11
MODEL_PATH  = os.path.join("models", "instrunet_cnn.h5")
LABELS_PATH = os.path.join("models", "label_classes.json")

CLASSES = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# =============================================
# BUILD CNN MODEL
# =============================================
def build_model(input_shape=(128, 128, 3), num_classes=11):
    inp = keras.Input(shape=input_shape, name="mel_spectrogram")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="InstruNet_CNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =============================================
# MAIN
# =============================================
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("[*] Building InstruNet CNN model...")
    model = build_model(input_shape=(*IMG_SIZE, 3), num_classes=N_CLASSES)
    model.summary()

    # Save model
    model.save(MODEL_PATH)
    print(f"[OK] Model saved -> {MODEL_PATH}")
    print(f"   Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")

    # Save label classes
    with open(LABELS_PATH, "w") as f:
        json.dump(CLASSES, f)
    print(f"[OK] Labels saved -> {LABELS_PATH}")

    # Quick verify
    loaded = tf.keras.models.load_model(MODEL_PATH)
    dummy  = np.random.rand(1, 128, 128, 3).astype(np.float32)
    pred   = loaded.predict(dummy, verbose=0)
    print(f"\n[OK] Verification passed - output shape: {pred.shape}, sum: {pred.sum():.4f}")
    print("\n[NEXT] Now run:")
    print('   git add models/instrunet_cnn.h5 models/label_classes.json')
    print('   git commit -m "feat: add InstruNet CNN model"')
    print('   git push origin main')
