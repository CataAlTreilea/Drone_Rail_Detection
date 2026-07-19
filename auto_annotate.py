"""
Semi-automatic annotation tool.

Runs the detection model (exp.pt) on every image in a folder, shows you
the predicted bounding boxes, and lets you accept or discard each one.
Accepted frames are saved with a YOLO-format .txt label file so they can
be used directly for retraining with train.py.

Usage:
    python auto_annotate.py --images Dataset/train/Drone
    python auto_annotate.py --images Dataset/train/Drone --conf 0.25

Controls (shown on each frame):
    A  – Accept   → saves the .txt label file next to the image
    D  – Discard  → skips this image (no label saved)
    Q  – Quit     → stops and prints a summary

Classes:
    0 = pad
    1 = drone
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "exp.pt"
CLASS_NAMES = {0: "pad", 1: "drone"}
COLORS      = {0: (0, 165, 255), 1: (0, 255, 0)}   # pad=orange, drone=green

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _draw(frame: np.ndarray, boxes) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    for box in boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(cls, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls, cls)}  {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Controls legend
    legend = ["[A] Accept", "[D] Discard", "[Q] Quit"]
    for i, txt in enumerate(legend):
        cv2.putText(out, txt, (10, h - 15 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, txt, (10, h - 15 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return out


def _to_yolo_line(box, img_w: int, img_h: int) -> str:
    cls = int(box.cls[0])
    x1, y1, x2, y2 = map(float, box.xyxy[0])
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def run(img_dir: Path, conf_thresh: float):
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {img_dir}")
        return

    model = YOLO(MODEL_PATH)
    device = 0
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    accepted = 0
    discarded = 0
    skipped = 0   # no detections above threshold

    print(f"Found {len(images)} images in '{img_dir}'")
    print("Controls: A = accept | D = discard | Q = quit\n")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [WARN] Could not read {img_path.name}, skipping")
            continue

        h, w = frame.shape[:2]
        result = model.predict(frame, device=device, verbose=False, conf=conf_thresh, half=(device != "cpu"))[0]
        boxes = result.boxes

        progress = f"[{idx + 1}/{len(images)}]"

        if boxes is None or len(boxes) == 0:
            # No detections — show the plain frame, allow accept (blank label) or discard
            display = frame.copy()
            cv2.putText(display, "No detections", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            legend = ["[A] Accept (empty label)", "[D] Discard", "[Q] Quit"]
            for i, txt in enumerate(legend):
                cv2.putText(display, txt, (10, h - 15 - i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, txt, (10, h - 15 - i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow(f"Auto-Annotate  {progress}  {img_path.name}", display)
        else:
            display = _draw(frame, boxes)
            n = len(boxes)
            cv2.putText(display, f"{progress}  {img_path.name}  ({n} detection{'s' if n != 1 else ''})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.imshow(f"Auto-Annotate  {progress}  {img_path.name}", display)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('a'), ord('A')):
                # Save label file
                lines = [_to_yolo_line(b, w, h) for b in boxes] if boxes and len(boxes) > 0 else []
                label_path = img_path.with_suffix(".txt")
                label_path.write_text("\n".join(lines))
                print(f"  ✓  {img_path.name}  → saved {len(lines)} box(es)")
                accepted += 1
                break
            elif key in (ord('d'), ord('D')):
                print(f"  ✗  {img_path.name}  → discarded")
                discarded += 1
                break
            elif key in (ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                print(f"\nStopped early at image {idx + 1}/{len(images)}")
                _print_summary(accepted, discarded, skipped)
                return

        cv2.destroyAllWindows()

    _print_summary(accepted, discarded, skipped)


def _print_summary(accepted, discarded, skipped):
    total = accepted + discarded + skipped
    print(f"\n── Summary ──────────────────────────────")
    print(f"  Accepted : {accepted}")
    print(f"  Discarded: {discarded}")
    if skipped:
        print(f"  Skipped  : {skipped}  (no detections)")
    print(f"  Total    : {total}")
    print(f"\nAccepted images now have .txt label files alongside them.")
    print("Run  python train.py --model det  to retrain the detection model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-automatic YOLO annotation with per-frame review")
    parser.add_argument("--images", type=Path, default=Path("Dataset/train/Drone"),
                        help="Folder of images to annotate (default: Dataset/train/Drone)")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Detection confidence threshold (default: 0.20, lower = more detections)")
    args = parser.parse_args()

    if not args.images.exists():
        print(f"Error: folder not found: {args.images}")
        exit(1)

    run(args.images, args.conf)
