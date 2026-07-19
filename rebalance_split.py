"""
Rebalance drone train/validation split to ~80% train / 20% val.

Currently: 19 train, 64 val  →  Target: ~66 train, ~17 val
Moves images (and matching label files if present) from validation to train.

Run once:
    python rebalance_split.py
"""

import random
import shutil
from pathlib import Path

TRAIN_DIR   = Path("Dataset") / "train"   / "Drone"
VAL_DIR     = Path("Dataset") / "validation" / "Drone"
TRAIN_RATIO = 0.80
SEED        = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}


def rebalance():
    train_imgs = sorted(p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    val_imgs   = sorted(p for p in VAL_DIR.iterdir()   if p.suffix.lower() in IMAGE_EXTENSIONS)

    total   = len(train_imgs) + len(val_imgs)
    target_train = int(total * TRAIN_RATIO)
    need    = max(0, target_train - len(train_imgs))

    print(f"Total images : {total}")
    print(f"Current split: {len(train_imgs)} train / {len(val_imgs)} val")
    print(f"Target split : {target_train} train / {total - target_train} val")
    print(f"Moving       : {need} images from val → train")

    if need == 0:
        print("Already balanced — nothing to do.")
        return

    random.seed(SEED)
    to_move = random.sample(val_imgs, need)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    for src in to_move:
        dst = TRAIN_DIR / src.name
        shutil.move(str(src), str(dst))
        print(f"  moved {src.name}")

        # Move matching label file if it exists (YOLO .txt labels)
        label_src = src.with_suffix(".txt")
        if label_src.exists():
            label_dst = TRAIN_DIR / label_src.name
            shutil.move(str(label_src), str(label_dst))

    new_train = len(list(TRAIN_DIR.iterdir()))
    new_val   = len(list(VAL_DIR.iterdir()))
    print(f"\nDone. New split: {new_train} train / {new_val} val")


if __name__ == "__main__":
    rebalance()
