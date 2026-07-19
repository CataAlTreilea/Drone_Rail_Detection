"""
Retrain / fine-tune both models.

  Segmentation model (segment.pt)  – rails
      Converts RailsLabels/annotations.xml (CVAT polygon format) to YOLO
      segmentation labels, then fine-tunes from the existing segment.pt.

  Detection model (exp.pt)  – drone + pad
      Trains from exp.pt once you have bounding-box labels in YOLO format.
      See the LABELING INSTRUCTIONS section below for how to create them.

Run:
    python train.py --model seg      # retrain segmentation only
    python train.py --model det      # retrain detection only
    python train.py --model all      # retrain both (default)

-------------------------------------------------------------------------------
LABELING INSTRUCTIONS (detection model)
-------------------------------------------------------------------------------
You need one .txt label file per image in:
    Dataset/train/Drone/       <- images of the drone / pad
    Dataset/validation/Drone/

Each .txt file must have the same stem as the image and contain one line per
object in YOLO format:
    <class_id> <cx> <cy> <w> <h>        (all values 0-1, relative to image size)

Classes:
    0 = pad
    1 = drone

Recommended free labeling tools:
    • labelme  (already in requirements.txt)  ->  run: labelme
    • CVAT online  https://app.cvat.ai
    • Roboflow   https://roboflow.com

Once labels are ready, run:
    python train.py --model det
-------------------------------------------------------------------------------
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT           = Path(".")
CVAT_XML       = ROOT / "RailsLabels" / "annotations.xml"
RAILS_IMG_DIR  = ROOT / "Dataset" / "train" / "Rails"
SEG_WORK_DIR   = ROOT / "Dataset" / "seg_training"          # temp working dir for seg

SEG_MODEL_PATH = ROOT / "segment.pt"
DET_MODEL_PATH = ROOT / "exp.pt"

SEG_OUT        = ROOT / "runs" / "segment"
DET_OUT        = ROOT / "runs" / "detect"

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS    = 50
IMGSZ     = 640
BATCH     = 8       # lower to 4 if you get out-of-memory errors
DEVICE    = 0       # 0 = first GPU; "cpu" to force CPU


# ── Segmentation helpers ─────────────────────────────────────────────────────

def _cvat_to_yolo_seg(xml_path: Path, img_src_dir: Path, work_dir: Path) -> Path:
    """
    Convert CVAT polygon annotations to YOLO segmentation label files.
    Copies matching images alongside the labels.
    Returns path to the generated dataset YAML.
    """
    print("[SEG] Converting CVAT XML → YOLO segmentation labels …")

    train_img   = work_dir / "images" / "train"
    train_lbl   = work_dir / "labels" / "train"
    val_img     = work_dir / "images" / "val"
    val_lbl     = work_dir / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    tree  = ET.parse(xml_path)
    root  = tree.getroot()
    images = root.findall("image")

    # 80 / 20 split
    split_idx = max(1, int(len(images) * 0.8))
    train_set = images[:split_idx]
    val_set   = images[split_idx:]

    def _write_split(subset, img_out, lbl_out):
        written = 0
        for img_elem in subset:
            fname  = img_elem.attrib["name"]
            W      = float(img_elem.attrib["width"])
            H      = float(img_elem.attrib["height"])
            src    = img_src_dir / fname

            if not src.exists():
                print(f"  [WARN] image not found, skipping: {fname}")
                continue

            lines = []
            for poly in img_elem.findall("polygon"):
                # class 0 = Rails
                pts = poly.attrib["points"].split(";")
                norm = []
                for i, pt in enumerate(pts):
                    x, y = map(float, pt.strip().split(","))
                    norm += [f"{x / W:.6f}", f"{y / H:.6f}"]
                lines.append("0 " + " ".join(norm))

            if not lines:
                continue

            shutil.copy(src, img_out / fname)
            (lbl_out / Path(fname).with_suffix(".txt").name).write_text("\n".join(lines))
            written += 1

        return written

    n_train = _write_split(train_set, train_img, train_lbl)
    n_val   = _write_split(val_set,   val_img,   val_lbl)
    print(f"  → {n_train} train / {n_val} val images prepared")

    yaml_path = work_dir / "rails_seg.yaml"
    yaml_path.write_text(
        f"path: {work_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "names:\n"
        "  0: Rails\n"
    )
    return yaml_path


def train_segmentation():
    print("\n══ SEGMENTATION MODEL ══════════════════════════════════════════")

    yaml_path = _cvat_to_yolo_seg(CVAT_XML, RAILS_IMG_DIR, SEG_WORK_DIR)

    model = YOLO(str(SEG_MODEL_PATH))   # fine-tune from existing weights
    model.train(
        data    = str(yaml_path),
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        half    = True,                 # FP16 – faster on RTX 4070
        project = str(SEG_OUT),
        name    = "fine_tune",
        exist_ok= True,
        patience= 15,                   # early stop if no improvement for 15 epochs
        augment = True,
    )

    best = SEG_OUT / "fine_tune" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, ROOT / "segment.pt")
        print(f"\n[SEG] ✓ New weights saved to segment.pt  (was {SEG_MODEL_PATH})")
    else:
        print("[SEG] Training finished but best.pt not found — check runs/segment/")


# ── Detection helpers ─────────────────────────────────────────────────────────

def _build_det_yaml() -> Path | None:
    """
    Build detection dataset YAML from Dataset/train/Drone and Dataset/validation/Drone.
    Returns None if no label files are found (labels not yet created).
    """
    train_labels = list((ROOT / "Dataset" / "train" / "Drone").glob("*.txt"))
    val_labels   = list((ROOT / "Dataset" / "validation" / "Drone").glob("*.txt"))

    if not train_labels:
        return None

    yaml_path = ROOT / "Dataset" / "drone_det.yaml"
    yaml_path.write_text(
        f"path: {(ROOT / 'Dataset').resolve()}\n"
        "train: train/Drone\n"
        "val:   validation/Drone\n"
        "names:\n"
        "  0: pad\n"
        "  1: drone\n"
    )
    return yaml_path


def train_detection():
    print("\n══ DETECTION MODEL ═════════════════════════════════════════════")

    yaml_path = _build_det_yaml()
    if yaml_path is None:
        print(
            "[DET] ✗  No label files found in Dataset/train/Drone/\n"
            "         Label your images first — see the LABELING INSTRUCTIONS\n"
            "         at the top of this file, then re-run:\n"
            "             python train.py --model det"
        )
        return

    model = YOLO(str(DET_MODEL_PATH))   # fine-tune from existing weights
    model.train(
        data    = str(yaml_path),
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        half    = True,
        project = str(DET_OUT),
        name    = "fine_tune",
        exist_ok= True,
        patience= 15,
        augment = True,
    )

    best = DET_OUT / "fine_tune" / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, ROOT / "exp.pt")
        print(f"\n[DET] ✓ New weights saved to exp.pt  (was {DET_MODEL_PATH})")
    else:
        print("[DET] Training finished but best.pt not found — check runs/detect/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["seg", "det", "all"],
        default="all",
        help="Which model to retrain (default: all)",
    )
    args = parser.parse_args()

    if args.model in ("seg", "all"):
        train_segmentation()
    if args.model in ("det", "all"):
        train_detection()
