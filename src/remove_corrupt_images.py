# src/remove_corrupt_images.py

"""
Standalone script to remove corrupt images from a given folder tree.
By default, it targets data/flattened/, but you can edit FOLDER_TO_CHECK as needed.
"""

import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# === CONFIGURATION ===
FOLDER_TO_CHECK = Path("data/flattened")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CORRUPT_LOG = Path("data/flattened/corrupt_images.txt")


def remove_corrupt_images(root: Path):
    corrupts = []
    for sub in root.iterdir():
        if sub.is_dir():
            for img_file in sub.iterdir():
                if img_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                try:
                    with Image.open(img_file) as img:
                        img.verify()
                except (UnidentifiedImageError, IOError, OSError):
                    corrupts.append(str(img_file))
                    img_file.unlink()
    if corrupts:
        with open(CORRUPT_LOG, "w") as f:
            for path in corrupts:
                f.write(path + "\n")
        print(f"Deleted {len(corrupts)} corrupt images. See {CORRUPT_LOG}")
    else:
        print("No corrupt images found.")


if __name__ == "__main__":
    remove_corrupt_images(FOLDER_TO_CHECK)
