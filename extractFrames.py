"""
Extract every Nth frame from a video file.

Edit the parameters below, then run:
    python extract_frames.py
"""

import os
import cv2
from pathlib import Path

# -------- Parameters --------
VIDEO_PATH = Path(".") / "Dataset" / "Video.mp4"   # Path to the input video
OUTPUT_DIR = Path(".") / "Dataset" / "train" / "Rails"         # Directory where extracted frames will be saved
STEP = 10                     # Save one frame every STEP frames
# ----------------------------


def extract_frames(video_path: str, output_dir: str, step: int) -> int:
    """
    Extract every `step`-th frame from `video_path` and save as PNG files
    in `output_dir`. Returns the number of frames saved.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pad filenames so they sort naturally (frame_00000.png, frame_00001.png, ...)
    pad = max(5, len(str(total)))

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            out_path = os.path.join(output_dir, f"frame_{frame_idx:0{pad}d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Read {frame_idx} frames, saved {saved} to '{output_dir}' (every {step}th frame).")
    return saved


if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, STEP)