import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO


DRONE_MODEL_PATH = "exp.pt"
SEGMENT_MODEL_PATH = "segment.pt"

CLASS_PAD = 0
CLASS_DRONE = 1

# Pad bbox area threshold (pixels²) to confirm endpoint reached
PAD_AREA_THRESHOLD = 5000


@dataclass
class FrameResult:
    drone_center: tuple[int, int] | None        # (x, y) pixel center of drone
    pad_detected: bool                           # True when pad visible and large enough
    pad_center: tuple[int, int] | None           # (x, y) pixel center of pad
    rail_centerline_x: int | None                # x-coord of rail centroid in frame
    lateral_offset: int | None                   # drone_center.x - rail_centerline_x (+ = right)
    vertical_offset: int | None                  # drone_center.y - rail_centerline_y (+ = below)
    path_points: list[tuple[int, int]] | None    # smoothed flight path along rail centerline
    annotated_frame: np.ndarray                  # BGR frame with all overlays drawn


class FrameGrabber:
    """Background thread that continuously drains the camera buffer so the
    main loop always gets the latest frame instead of a stale one."""

    def __init__(self, url: str):
        self._url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._ret = False
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None
            if not ret or frame is None:
                try:
                    self.cap.open(self._url, cv2.CAP_FFMPEG)
                except Exception:
                    pass
                continue
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        self._stop = True
        self.cap.release()


def _load_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)


def load_models() -> tuple[str | int, YOLO, YOLO]:
    print(f"torch {torch.__version__}")
    print(f"cuda {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else "cpu"
    drone_model = _load_model(DRONE_MODEL_PATH)
    segment_model = _load_model(SEGMENT_MODEL_PATH)
    return device, drone_model, segment_model


def _get_best_mask(seg_result) -> np.ndarray | None:
    """Return the largest rail mask resized to the original frame size, or None."""
    if seg_result.masks is None or len(seg_result.masks) == 0:
        return None

    best_mask, best_area = None, 0
    for mask in seg_result.masks.data:
        m = mask.cpu().numpy().astype(np.uint8)
        area = int(m.sum())
        if area > best_area:
            best_area, best_mask = area, m

    if best_mask is None:
        return None

    h, w = seg_result.orig_shape
    return cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _rail_centroid(seg_result) -> tuple[int, int] | None:
    """Return (cx, cy) centroid of largest rail mask, or None."""
    mask = _get_best_mask(seg_result)
    if mask is None:
        return None
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _rail_path(mask: np.ndarray, sample_step: int = 4) -> list[tuple[int, int]] | None:
    """
    Compute a flight path along the rail centerline.
    Samples the vertical centre of the mask for every `sample_step` columns,
    then fits a lightly-smoothed cubic spline that stays close to the actual
    rail shape rather than averaging it away.
    Returns a list of (x, y) pixel tuples, or None if the mask is too sparse.
    """
    h, w = mask.shape
    xs, ys = [], []
    for x in range(0, w, sample_step):
        col_ys = np.where(mask[:, x] > 0)[0]
        if len(col_ys) >= 3:
            xs.append(x)
            ys.append(int(col_ys.mean()))

    if len(xs) < 5:
        return None

    try:
        # s = len(xs) * 3 keeps the spline tight to the sampled points
        # while still removing single-pixel noise
        spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * 3)
        px = np.linspace(xs[0], xs[-1], 300, dtype=np.int32)
        py = np.clip(spl(px), 0, h - 1).astype(np.int32)
        return list(zip(px.tolist(), py.tolist()))
    except Exception:
        return None


class PathSmoother:
    """
    Temporal EMA smoother for the rail path, with stability detection.

    Each call to update() blends the newly detected path with the history,
    preventing the line from jumping between frames.

    alpha = 0.0  → line never moves (pure history)
    alpha = 1.0  → no smoothing (raw per-frame path)
    alpha = 0.2  → 80% history / 20% new  (good default)

    is_stable returns True once the path has converged and is no longer
    changing significantly — safe to use as a fly condition.
    """

    N_PTS             = 300
    STABILITY_WINDOW  = 20    # frames used to compute the rolling average
    STABILITY_THRESH  = 3.0   # pixels — max mean frame-to-frame change to be "stable"
    MIN_FRAMES        = 30    # minimum frames before stability can be declared

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._fixed_px: np.ndarray | None = None
        self._smooth_py: np.ndarray | None = None
        self._frame_count: int = 0
        self._changes: list[float] = []   # rolling frame-to-frame deltas
        self._locked: bool = False        # when True, path is frozen permanently

    # ── public properties ─────────────────────────────────────────────────────

    @property
    def stability_score(self) -> float:
        """Mean frame-to-frame change (pixels). Lower = more stable."""
        if not self._changes:
            return float("inf")
        window = self._changes[-self.STABILITY_WINDOW:]
        return float(np.mean(window))

    @property
    def is_stable(self) -> bool:
        return (self._frame_count >= self.MIN_FRAMES and
                self.stability_score < self.STABILITY_THRESH)

    @property
    def is_locked(self) -> bool:
        return self._locked

    def lock(self):
        """Freeze the current path permanently. update() will return it unchanged."""
        self._locked = True

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, path_points: list[tuple[int, int]] | None,
               frame_w: int) -> list[tuple[int, int]] | None:
        # If locked, always return the frozen path — ignore new detections.
        if self._locked:
            if self._smooth_py is None or self._fixed_px is None:
                return None
            return list(zip(self._fixed_px.tolist(), self._smooth_py.astype(int).tolist()))

        # Lazy-initialise fixed x-grid once we know the frame width.
        if self._fixed_px is None or self._fixed_px[-1] != frame_w - 1:
            self._fixed_px = np.linspace(0, frame_w - 1, self.N_PTS, dtype=np.int32)
            self._smooth_py = None
            self._frame_count = 0
            self._changes = []

        if path_points is None:
            # No rail this frame — hold last known path.
            if self._smooth_py is None:
                return None
            return list(zip(self._fixed_px.tolist(), self._smooth_py.astype(int).tolist()))

        raw_px = np.array([p[0] for p in path_points], dtype=np.float32)
        raw_py = np.array([p[1] for p in path_points], dtype=np.float32)
        new_py = np.interp(self._fixed_px, raw_px, raw_py).astype(np.float32)

        if self._smooth_py is None:
            self._smooth_py = new_py
        else:
            prev = self._smooth_py.copy()
            self._smooth_py = self.alpha * new_py + (1.0 - self.alpha) * self._smooth_py
            self._changes.append(float(np.abs(self._smooth_py - prev).mean()))
            if len(self._changes) > self.STABILITY_WINDOW * 2:
                self._changes = self._changes[-self.STABILITY_WINDOW:]

        self._frame_count += 1
        return list(zip(self._fixed_px.tolist(), self._smooth_py.astype(int).tolist()))


def process_frame(
    frame: np.ndarray,
    drone_model: YOLO,
    segment_model: YOLO,
    device: str | int,
    smoother: PathSmoother | None = None,
    detect_only: bool = False,
) -> FrameResult:
    """
    detect_only=True  → skip segmentation model entirely (path comes from
                         locked smoother). Use during flight once path is set.
    detect_only=False → run both models (pre-flight, path building phase).
    """
    fp16 = device != "cpu"
    det_result = drone_model.predict(frame, device=device, verbose=False, half=fp16)[0]

    # ── Segmentation (skipped during flight) ─────────────────────────────────
    seg_result = None
    if not detect_only:
        seg_result = segment_model.predict(frame, device=device, verbose=False, half=fp16)[0]

    # ── Detection boxes ───────────────────────────────────────────────────────
    drone_center: tuple[int, int] | None = None
    pad_detected = False
    pad_center: tuple[int, int] | None = None

    for box in det_result.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cls == CLASS_DRONE:
            drone_center = (cx, cy)
        elif cls == CLASS_PAD:
            area = (x2 - x1) * (y2 - y1)
            pad_center = (cx, cy)
            pad_detected = area >= PAD_AREA_THRESHOLD

    # ── Rail mask → path ──────────────────────────────────────────────────────
    h_frame, w_frame = frame.shape[:2]
    rail_cx, rail_cy, rail_centroid = None, None, None
    path_points: list[tuple[int, int]] | None = None

    if detect_only:
        # Path comes entirely from the locked smoother
        raw_path = None
    else:
        rail_mask = _get_best_mask(seg_result)
        if rail_mask is not None:
            moments = cv2.moments(rail_mask)
            if moments["m00"] != 0:
                rail_cx = int(moments["m10"] / moments["m00"])
                rail_cy = int(moments["m01"] / moments["m00"])
                rail_centroid = (rail_cx, rail_cy)
            raw_path = _rail_path(rail_mask)
        else:
            raw_path = None

    if smoother is not None:
        path_points = smoother.update(raw_path, w_frame)
    else:
        path_points = raw_path

    lateral_offset = None
    vertical_offset = None
    if drone_center is not None and rail_cx is not None:
        lateral_offset = drone_center[0] - rail_cx
    if drone_center is not None and rail_cy is not None:
        vertical_offset = drone_center[1] - rail_cy

    # ── Draw overlays ─────────────────────────────────────────────────────────
    out = det_result.plot()

    if seg_result is not None:
        seg_mask_img = seg_result.plot()
        out = cv2.addWeighted(out, 0.7, seg_mask_img, 0.3, 0)

    # Flight path — red curve
    if path_points is not None and len(path_points) > 1:
        pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=False, color=(0, 0, 255), thickness=3)

    if rail_centroid is not None:
        cv2.circle(out, rail_centroid, 8, (255, 100, 0), -1)
        cv2.putText(out, "rail", (rail_cx + 10, rail_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    if drone_center is not None:
        cv2.circle(out, drone_center, 8, (0, 255, 0), -1)
        if lateral_offset is not None:
            cv2.putText(out, f"off={lateral_offset:+d}px",
                        (drone_center[0] + 10, drone_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Line from drone to nearest point on the path
    if drone_center is not None and path_points is not None and len(path_points) > 1:
        px_arr = np.array([p[0] for p in path_points], dtype=np.float32)
        py_arr = np.array([p[1] for p in path_points], dtype=np.float32)
        dists  = np.sqrt((px_arr - drone_center[0]) ** 2 +
                         (py_arr - drone_center[1]) ** 2)
        closest = path_points[int(np.argmin(dists))]
        cv2.line(out, drone_center, closest, (0, 255, 255), 2)
        cv2.circle(out, closest, 7, (0, 255, 255), -1)

    if drone_center is not None and rail_centroid is not None:
        cv2.line(out, drone_center, rail_centroid, (255, 100, 0), 1)

    if pad_center is not None:
        color = (0, 0, 255) if pad_detected else (0, 165, 255)
        cv2.putText(out, "PAD-STOP" if pad_detected else "pad",
                    (pad_center[0] + 10, pad_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return FrameResult(
        drone_center=drone_center,
        pad_detected=pad_detected,
        pad_center=pad_center,
        rail_centerline_x=rail_cx,
        lateral_offset=lateral_offset,
        vertical_offset=vertical_offset,
        path_points=path_points,
        annotated_frame=out,
    )
