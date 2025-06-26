# src/prepare_dataset.py

"""
1) Recursively traverse `data/raw/` to find all folders named like "*PPM".
2) Copy every image inside them into `data/flattened/<PPM_folder>/`.
3) Attempt to open each copied image with PIL. If it fails, delete it and record its name in corrupt_images.txt.
"""

import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# === CONFIGURATION ===
RAW_DIR = Path("data/raw")
FLATTENED_DIR = Path("data/flattened")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# (Optional) Log of corrupt images
CORRUPT_LOG = Path("data/flattened/corrupt_images.txt")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def find_ppm_folders(root: Path):
    """
    Recursively find every subfolder whose name ends with 'PPM' (case-insensitive).
    Return a list of (folder_path, folder_name).
    """
    ppm_folders = []
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            if dirname.lower().endswith("ppm"):
                ppm_folders.append((Path(dirpath) / dirname, dirname))
    return ppm_folders


def copy_and_flatten(ppm_folders):
    """
    For each (folder_path, folder_name) in ppm_folders,
    copy all images into FLATTENED_DIR/<folder_name>/
    """
    if not FLATTENED_DIR.exists():
        FLATTENED_DIR.mkdir(parents=True, exist_ok=True)

    for folder_path, ppm_name in ppm_folders:
        dest_dir = FLATTENED_DIR / ppm_name
        dest_dir.mkdir(exist_ok=True)
        for item in folder_path.iterdir():
            if item.is_file() and is_image_file(item):
                dest = dest_dir / item.name
                # If same filename already exists, rename by appending a number
                counter = 1
                while dest.exists():
                    dest = dest_dir / f"{item.stem}_{counter}{item.suffix}"
                    counter += 1
                shutil.copy2(item, dest)


def remove_corrupt_images():
    """
    Go through every file in data/flattened/*/*. Try to open with PIL.
    If it fails, delete the file and write its path to CORRUPT_LOG.
    """
    corrupts = []
    for ppm_folder in FLATTENED_DIR.iterdir():
        if ppm_folder.is_dir():
            for img_file in ppm_folder.iterdir():
                if not is_image_file(img_file):
                    continue
                try:
                    with Image.open(img_file) as img:
                        img.verify()  # verify does not decode entire image but checks integrity
                except (UnidentifiedImageError, IOError, OSError) as e:
                    corrupts.append(str(img_file))
                    img_file.unlink()  # delete corrupt image

    if corrupts:
        with open(CORRUPT_LOG, "w") as f:
            for path in corrupts:
                f.write(path + "\n")
        print(f"Found and deleted {len(corrupts)} corrupt images. See {CORRUPT_LOG}")
    else:
        print("No corrupt images found.")


if __name__ == "__main__":
    print("Finding all PPM‐labeled folders under data/raw/ …")
    ppm_folders = find_ppm_folders(RAW_DIR)
    print(f"→ Found {len(ppm_folders)} PPM‐folders.")

    print("Copying images to data/flattened/<PPM>/ …")
    copy_and_flatten(ppm_folders)
    print("Flattening complete.")

    print("Checking for corrupt images …")
    remove_corrupt_images()
    print("Dataset preparation done.")
