"""
Rail path preview.

Runs the segmentation model on a video, image folder, or live camera stream,
computes a smooth flight path along the rail centerline, and draws it in red
so you can visually confirm it looks correct.

What is drawn:
  - Semi-transparent green overlay  →  segmented rail mask
  - Blue dots                       →  raw sampled centerline points
  - Red curve                       →  smoothed drone flight path
  - Yellow dashed line              →  target flight altitude offset above rail

Usage:
    python path_preview.py --video  Dataset/Video.mp4
    python path_preview.py --images Dataset/train/Rails
    python path_preview.py --live                          # phone camera stream

Controls:
    Q  – quit
    S  – save current frame as PNG
    +  – raise flight altitude (increases vertical offset)
    -  – lower flight altitude
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
SEGMENT_MODEL_PATH = "segment.pt"
CAMERA_SERVER      = "https://192.168.1.133:8080/video"

# How far above the rail centerline the drone path is drawn (pixels).
# Positive = upward in frame (smaller y). Adjust interactively with +/-.
DEFAULT_ALTITUDE_OFFSET = 80

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

SAVE_DIR = Path("path_preview_frames")


# ── Path generation ───────────────────────────────────────────────────────────

def _largest_mask(seg_result) -> np.ndarray | None:
    """Return the largest rail mask resized to the original frame, or None."""
    if seg_result.masks is None or len(seg_result.masks) == 0:
        return None

    best, best_area = None, 0
    for mask in seg_result.masks.data:
        m = mask.cpu().numpy().astype(np.uint8)
        area = int(m.sum())
        if area > best_area:
            best_area, best = area, m

    if best is None:
        return None

    h, w = seg_result.orig_shape
    return cv2.resize(best, (w, h), interpolation=cv2.INTER_NEAREST)


def _centerline_points(mask: np.ndarray, sample_step: int = 6):
    """
    Sample the vertical centroid of the rail mask for every `sample_step`
    columns across the frame. Returns (xs, ys) arrays of raw centre points.
    """
    h, w = mask.shape
    xs, ys = [], []
    for x in range(0, w, sample_step):
        col_ys = np.where(mask[:, x] > 0)[0]
        if len(col_ys) >= 3:          # ignore thin / noisy hits
            xs.append(x)
            ys.append(int(col_ys.mean()))
    return np.array(xs), np.array(ys)


def _smooth_path(xs: np.ndarray, ys: np.ndarray, n_pts: int = 300):
    """
    Fit a smoothing spline through (xs, ys) and return `n_pts` evenly spaced
    points along it. Returns (px, py) arrays, or (None, None) on failure.
    """
    if len(xs) < 5:
        return None, None
    try:
        # k=3 cubic spline; s controls smoothness (larger = smoother)
        spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * 20)
        px  = np.linspace(xs[0], xs[-1], n_pts, dtype=np.int32)
        py  = np.clip(spl(px), 0, 1e6).astype(np.int32)
        return px, py
    except Exception:
        return None, None


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, mask: np.ndarray,
                  raw_xs, raw_ys,
                  path_px, path_py,
                  altitude_offset: int,
                  altitude_adjusted_py=None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Rail mask — semi-transparent green
    green = np.zeros_like(out)
    green[mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(out, 0.75, green, 0.25, 0)

    # 2. Rail mask contour — solid green border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 0), 2)

    # 3. Raw centerline samples — blue dots
    for x, y in zip(raw_xs, raw_ys):
        cv2.circle(out, (int(x), int(y)), 2, (255, 150, 0), -1)

    # 4. Smoothed flight path — thick red curve
    if path_px is not None and len(path_px) > 1:
        pts = np.column_stack([path_px, path_py]).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=3)

    # 5. Target altitude path — yellow dashed line (offset above rail path)
    if altitude_adjusted_py is not None and path_px is not None:
        step = max(1, len(path_px) // 40)
        for i in range(0, len(path_px) - step, step * 2):
            p1 = (int(path_px[i]),      int(altitude_adjusted_py[i]))
            p2 = (int(path_px[i+step]), int(altitude_adjusted_py[i+step]))
            cv2.line(out, p1, p2, (0, 230, 230), 2)

    # 6. HUD text
    hud = [
        f"altitude offset: {altitude_offset:+d}px  (+/- to adjust)",
        f"centerline pts : {len(raw_xs)}",
        "Q=quit  S=save frame",
    ]
    for i, txt in enumerate(hud):
        cv2.putText(out, txt, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, txt, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out


def _no_detection_frame(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, "No rail detected", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    cv2.putText(out, "Q=quit  S=save frame", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

def _process_frame(frame, model, device, altitude_offset):
    """Run model + path generation on one frame. Returns annotated frame."""
    seg_result = model.predict(frame, device=device, verbose=False,
                               half=(device != "cpu"))[0]
    mask = _largest_mask(seg_result)

    if mask is None:
        return _no_detection_frame(frame)

    raw_xs, raw_ys = _centerline_points(mask)
    path_px, path_py = _smooth_path(raw_xs, raw_ys)

    alt_py = None
    if path_py is not None:
        h = frame.shape[0]
        alt_py = np.clip(path_py - altitude_offset, 0, h - 1)

    return _draw_overlay(frame, mask, raw_xs, raw_ys,
                         path_px, path_py, altitude_offset, alt_py)


def _run_loop(source, model, device, altitude_offset, window_title):
    saved = 0
    SAVE_DIR.mkdir(exist_ok=True)

    while True:
        if callable(source):
            ok, frame = source()
            if not ok or frame is None:
                continue
        else:
            ok, frame = source.read()
            if not ok:
                break

        annotated = _process_frame(frame, model, device, altitude_offset)
        cv2.imshow(window_title, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            fname = SAVE_DIR / f"frame_{int(time.time()*1000)}.png"
            cv2.imwrite(str(fname), annotated)
            saved += 1
            print(f"  Saved {fname}")
        elif key == ord('+') or key == ord('='):
            altitude_offset += 10
            print(f"  Altitude offset → {altitude_offset}px")
        elif key == ord('-'):
            altitude_offset = max(0, altitude_offset - 10)
            print(f"  Altitude offset → {altitude_offset}px")

    cv2.destroyAllWindows()
    if saved:
        print(f"\nSaved {saved} frame(s) to '{SAVE_DIR}/'")


def _run_images(img_dir: Path, model, device, altitude_offset):
    images = sorted(p for p in img_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(images)} images. Press any key to advance, Q to quit.")
    SAVE_DIR.mkdir(exist_ok=True)
    saved = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        annotated = _process_frame(frame, model, device, altitude_offset)
        title = f"Path Preview [{idx+1}/{len(images)}]  {img_path.name}"
        cv2.imshow(title, annotated)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                if saved:
                    print(f"\nSaved {saved} frame(s) to '{SAVE_DIR}/'")
                return
            elif key in (ord('s'), ord('S')):
                fname = SAVE_DIR / f"{img_path.stem}_path.png"
                cv2.imwrite(str(fname), annotated)
                saved += 1
                print(f"  Saved {fname}")
            elif key == ord('+') or key == ord('='):
                altitude_offset += 10
                print(f"  Altitude offset → {altitude_offset}px")
                annotated = _process_frame(frame, model, device, altitude_offset)
                cv2.imshow(title, annotated)
            elif key == ord('-'):
                altitude_offset = max(0, altitude_offset - 10)
                print(f"  Altitude offset → {altitude_offset}px")
                annotated = _process_frame(frame, model, device, altitude_offset)
                cv2.imshow(title, annotated)
            else:
                break   # any other key → next image

        cv2.destroyAllWindows()

    if saved:
        print(f"\nSaved {saved} frame(s) to '{SAVE_DIR}/'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rail path preview")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video",  type=Path, help="Path to a video file")
    group.add_argument("--images", type=Path, help="Folder of images")
    group.add_argument("--live",   action="store_true",
                       help="Use live phone camera stream")
    parser.add_argument("--offset", type=int, default=DEFAULT_ALTITUDE_OFFSET,
                        help=f"Starting altitude offset in pixels (default: {DEFAULT_ALTITUDE_OFFSET})")
    args = parser.parse_args()

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(SEGMENT_MODEL_PATH)

    if args.images:
        _run_images(args.images, model, device, args.offset)

    elif args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"Could not open video: {args.video}")
            exit(1)
        _run_loop(cap.read, model, device, args.offset, f"Path Preview — {args.video.name}")
        cap.release()

    else:
        # Live camera (default)
        cap = cv2.VideoCapture(CAMERA_SERVER, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("Could not open camera stream")
            exit(1)

        import threading
        _latest = {"ret": False, "frame": None, "lock": threading.Lock()}

        def _grabber():
            while True:
                ret, frame = cap.read()
                with _latest["lock"]:
                    _latest["ret"] = ret
                    _latest["frame"] = frame

        threading.Thread(target=_grabber, daemon=True).start()

        def _read():
            with _latest["lock"]:
                return _latest["ret"], (_latest["frame"].copy()
                                        if _latest["frame"] is not None else None)

        _run_loop(_read, model, device, args.offset, "Path Preview — Live")
        cap.release()
