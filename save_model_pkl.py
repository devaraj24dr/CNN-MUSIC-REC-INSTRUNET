# ============================================
# Save Trained Model as PKL for Streamlit Deployment
# Bundles: model weights, architecture config,
#   label encoder, preprocessing params, class labels,
#   evaluation results, and instrument name mapping
# ============================================

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import LabelEncoder

# ===============================
# 1. Paths
# ===============================
ROOT_DIR = r"D:\AI ONBOARDING ENGINE TT\cnn_music_instrument_recognition"
MODEL_DIR = os.path.join(ROOT_DIR, "models")
PTH_PATH = os.path.join(ROOT_DIR, "instrument_classifier_best.pth")
PKL_OUTPUT = os.path.join(MODEL_DIR, "instrument_classifier_full.pkl")

# ===============================
# 2. Hardcode class labels
# ===============================
class_names = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
print(f"Classes ({len(class_names)}): {class_names}")

# ===============================
# 3. Rebuild LabelEncoder
# ===============================
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

# ===============================
# 4. Load evaluation results (Mocked if missing)
# ===============================
eval_results = {}

# ===============================
# 5. Rebuild model architecture
# ===============================
device = torch.device("cpu")  # Save on CPU for portability

class CustomCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CustomCNN(num_classes=len(class_names))

# Load trained weights
model.load_state_dict(torch.load(PTH_PATH, map_location=device))
model.eval()
print("Model weights loaded successfully.")

# ===============================
# 6. Instrument name mapping
# ===============================
INSTRUMENT_NAMES = {
    "cel": "Cello",
    "cla": "Clarinet",
    "flu": "Flute",
    "gac": "Acoustic Guitar",
    "gel": "Electric Guitar",
    "org": "Organ",
    "pia": "Piano",
    "sax": "Saxophone",
    "tru": "Trumpet",
    "vio": "Violin",
    "voi": "Voice",
}

# ===============================
# 7. Preprocessing config
# ===============================
preprocessing_config = {
    "sample_rate": 22050,
    "duration_seconds": 3,
    "n_mels": 128,
    "hop_length": 512,
    "target_shape": (128, 128),
    "normalization": "z-score",
    "channels": 1,                     
    "power_to_db_ref": "np.max",
}

# ===============================
# 8. Model architecture config
# ===============================
architecture_config = {
    "backbone": "CustomCNN",
    "num_classes": len(class_names),
    "input_shape": (1, 128, 128),
}

# ===============================
# 9. Bundle everything into PKL
# ===============================
model_bundle = {
    # --- Model ---
    "model_state_dict": model.state_dict(),
    "architecture_config": architecture_config,

    # --- Labels & Encoding ---
    "class_names": class_names,
    "label_encoder": label_encoder,
    "instrument_names": INSTRUMENT_NAMES,

    # --- Preprocessing ---
    "preprocessing_config": preprocessing_config,

    # --- Evaluation Metrics ---
    "evaluation_results": eval_results,

    # --- Metadata ---
    "metadata": {
        "framework": "PyTorch",
        "model_name": "InstruNet-CustomCNN",
        "version": "1.0",
        "source_pth": "instrument_classifier_best.pth",
    }
}

# ===============================
# 10. Save PKL
# ===============================
with open(PKL_OUTPUT, "wb") as f:
    pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_mb = os.path.getsize(PKL_OUTPUT) / (1024 * 1024)
print(f"\n{'=' * 55}")
print(f"  MODEL BUNDLE SAVED SUCCESSFULLY")
print(f"{'=' * 55}")
print(f"  Output: {PKL_OUTPUT}")
print(f"  Size:   {file_size_mb:.2f} MB")
print(f"  Keys:   {list(model_bundle.keys())}")
print(f"{'=' * 55}")

# ===============================
# 11. Verification - Load it back
# ===============================
print("\nVerifying PKL file...")
with open(PKL_OUTPUT, "rb") as f:
    loaded = pickle.load(f)

print(f"  ✅ Keys found:       {list(loaded.keys())}")
print(f"  ✅ Classes:          {loaded['class_names']}")
print(f"  ✅ Preprocessing:    {loaded['preprocessing_config']}")
print(f"  ✅ Architecture:     {loaded['architecture_config']['backbone']}")
print(f"  ✅ Num classes:      {loaded['architecture_config']['num_classes']}")
print(f"  ✅ Instrument map:   {len(loaded['instrument_names'])} instruments")
print(f"  ✅ Has eval results: {bool(loaded['evaluation_results'])}")

# Quick model rebuild test
test_model = models.efficientnet_b0(weights=None)
test_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, loaded['architecture_config']['num_classes'])
)
test_model.load_state_dict(loaded['model_state_dict'])
test_model.eval()
print(f"  ✅ Model rebuilt from PKL successfully!")
print(f"\nDone. You can now use '{os.path.basename(PKL_OUTPUT)}' for Streamlit deployment.")