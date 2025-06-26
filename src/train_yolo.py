# src/train_yolo.py

import os
from ultralytics import YOLO

# === CONFIGURATION ===
DATA_YAML = "data.yaml"
PRETRAINED_BACKBONE = "yolov8n.pt"     # Try 'yolov8s.pt' if you have GPU memory
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 20
LR0 = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT = 0.1
PROJECT_DIR = "models"
NAME = "test_tube_detector"
WEIGHTS_OUTPUT = os.path.join(PROJECT_DIR, "yolo_weights.pt")

# ✅ Moderate augmentation to prevent overfitting
AUGMENTATION_PARAMS = {
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.3,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.8,
    "mixup": 0.0,
}

os.makedirs(PROJECT_DIR, exist_ok=True)

def main():
    # Load model
    model = YOLO(PRETRAINED_BACKBONE)

    print(f"🚀 Starting YOLOv8 training for {EPOCHS} epochs …")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=NAME,
        exist_ok=True,
        patience=PATIENCE,
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT,
        cos_lr=True,               # cosine LR decay for smoother training
        save=True,                 # Save model every epoch
        verbose=True,              # More logging
        val=True,                  # Always run validation
        **AUGMENTATION_PARAMS
    )

    # Save best weights
    weights_dir = os.path.join(PROJECT_DIR, NAME, "weights")
    best_weights_path = os.path.join(weights_dir, "best.pt")
    if not os.path.isfile(best_weights_path):
        raise FileNotFoundError(f"❌ best.pt not found in {weights_dir}")
    os.replace(best_weights_path, WEIGHTS_OUTPUT)
    print(f"✅ Best YOLOv8 weights saved to {WEIGHTS_OUTPUT}")

if __name__ == "__main__":
    main()
