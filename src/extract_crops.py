# src/extract_crops.py

from pathlib import Path
import cv2
from ultralytics import YOLO
import re

# === CONFIGURATION ===
YOLO_WEIGHTS = r"models/test_tube_detector/weights/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = "cpu"

FLATTENED_DIR = Path("data/flattened")
CROPPED_DIR = Path("data/cropped")
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_folder_name(folder_name: str) -> str:
    """
    Extract numeric value from folder name and format it as <float>_PPM
    e.g., "0.2ppm", " 0_2 PPM", "0.2_PPM" → "0.2_PPM"
    """
    folder_name = folder_name.lower().replace("ppm", "").replace("_", ".").replace(" ", "").strip()
    match = re.search(r"(\d+(\.\d+)?)", folder_name)
    if not match:
        raise ValueError(f"Cannot parse numeric value from folder name: '{folder_name}'")
    ppm_value = float(match.group(1))
    return f"{ppm_value}_PPM"


def crop_and_save(image_path: Path, box, output_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Failed to read image: {image_path}")
        return

    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 10 or y2 - y1 < 10:
        print(f"⚠️ Skipped too-small crop from {image_path}")
        return

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"⚠️ Empty crop from {image_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / image_path.name
    cv2.imwrite(str(dest_path), crop)
    print(f"✅ Saved crop: {dest_path}")


def main():
    model = YOLO(YOLO_WEIGHTS)
    model.fuse()

    for ppm_folder in FLATTENED_DIR.iterdir():
        if not ppm_folder.is_dir():
            continue

        try:
            ppm_name = normalize_folder_name(ppm_folder.name)
        except ValueError as e:
            print(f"❌ Skipping folder due to error: {e}")
            continue

        out_ppm_dir = CROPPED_DIR / ppm_name
        out_ppm_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📂 Processing folder: {ppm_name}")

        for img_file in ppm_folder.glob("*"):
            if not img_file.is_file() or img_file.suffix.lower() not in VALID_IMAGE_EXT:
                continue

            results = model.predict(
                source=str(img_file),
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

            if len(results) == 0 or results[0].boxes.data.shape[0] == 0:
                print(f"❌ No detection in {img_file.name}")
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            print(f"📷 {img_file.name} → Detected {len(boxes)} boxes (classes: {set(classes)})")

            # Save the most confident box
            top_idx = confs.argmax()
            best_box = boxes[top_idx]
            crop_and_save(img_file, best_box, out_ppm_dir)

    print("\n✅ All crops saved under data/cropped/<PPM_folder>.")


if __name__ == "__main__":
    main()
