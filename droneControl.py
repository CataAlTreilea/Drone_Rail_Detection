"""
Autonomous rail-following flight controller.

── Phase 1: Pre-flight ──────────────────────────────────────────────────────
  Both models run. Waits until:
    • Path is stable (smoother converged)
    • Drone is visible in frame
  → Path is locked. Segmentation model stops.

── Phase 2: Target selection ────────────────────────────────────────────────
  Frame is frozen on the moment the drone is confirmed.
  Click anywhere on the red path to set the stop point.
  Press ENTER to confirm, Q to abort.

── Phase 3: Flight ──────────────────────────────────────────────────────────
  Detection model only (no segmentation).
  Drone follows the locked red path toward the clicked target.
  Stops when:
    • Drone reaches target x position
    • Height exceeds MAX_HEIGHT_CM (emergency land)
    • FLIGHT_DURATION_SEC elapsed (safety timeout)
    • Pad detected
    • Q pressed

Frame axes (side camera):
  x : left → right  = forward along rail
  y : top  → bottom = inverse of height
"""

import os
import time

import cv2
import numpy as np
from codrone_edu.drone import Drone

import railDetection

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "flight_plan.png")

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_SERVER = "https://192.168.1.133:8080/video"

# ── Flight parameters ─────────────────────────────────────────────────────────
FORWARD_PITCH        = 40     # % pitch forward along rail — constant, always applied
BASE_THROTTLE        = 0
TAKEOFF_HOVER_SEC    = 2.0
FLIGHT_DURATION_SEC  = 8.0    # hard safety timeout — land if target not reached

# ── Height limit ──────────────────────────────────────────────────────────────
MAX_HEIGHT_CM        = 150    # 1.5 metres — emergency land if exceeded

# ── Height controller ─────────────────────────────────────────────────────────
# Drone tracks the red path y directly — no pixel offset.
# Positive error (drone below path in frame) → throttle up.
THROTTLE_KP          = 0.30

# ── Safety ────────────────────────────────────────────────────────────────────
THROTTLE_MAX         = 60
THROTTLE_MIN         = -60
TARGET_REACH_PX      = 40    # pixels — how close counts as "reached target"
COMMAND_INTERVAL_SEC = 0.05  # send commands at 20 Hz, not every camera frame


# ── Control helpers ───────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> int:
    return int(max(lo, min(hi, v)))


def _compute_controls(drone_center, path_points, pitch_cmd: int = FORWARD_PITCH):
    """
    Returns (pitch, throttle, dist_to_path).

    pitch_cmd : signed pitch to apply (caller decides direction).
    throttle  : P-controller that keeps drone on the path's y at its x position.
                drone below path line → climb; drone above → descend.
    dist      : Euclidean distance in pixels from drone to nearest path point.
    """
    if drone_center is None or path_points is None:
        return pitch_cmd, BASE_THROTTLE, None

    drone_x, drone_y = drone_center
    px = np.array([p[0] for p in path_points], dtype=np.float32)
    py = np.array([p[1] for p in path_points], dtype=np.float32)

    # Distance to nearest path point
    dists = np.sqrt((px - drone_x) ** 2 + (py - drone_y) ** 2)
    dist  = float(dists[int(np.argmin(dists))])

    # Throttle: drive drone_y toward path_y at drone's x
    # In frame: large y = low, small y = high.
    # drone_y > path_y → drone is BELOW the line → need to climb (positive throttle)
    path_y  = float(np.interp(drone_x, px, py))
    h_error = drone_y - path_y          # + = below path = climb
    throttle = _clamp(THROTTLE_KP * h_error + BASE_THROTTLE,
                     THROTTLE_MIN, THROTTLE_MAX)

    return pitch_cmd, throttle, dist


def _get_height_cm(drone: Drone) -> float | None:
    """Return drone height in cm, or None if sensor unavailable."""
    try:
        h = drone.get_height()
        return float(h) if h is not None else None
    except Exception:
        return None


# ── Phase 2: target selection ─────────────────────────────────────────────────

def _select_target(frozen_frame: np.ndarray,
                   path_points: list | None) -> tuple[int, int] | None:
    """
    Show the frozen frame and let the user click a stop point on the path.
    The click is snapped to the nearest path point.
    Returns (x, y) in pixel coords, or None if aborted.
    """
    selected  = [None]
    displays  = [None]

    def _redraw(pt=None):
        img = frozen_frame.copy()
        # Re-draw path
        if path_points and len(path_points) > 1:
            pts = np.array(path_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 0, 255), 3)
        # Draw selected point
        if pt is not None:
            cv2.circle(img, pt, 14, (0, 255, 255), -1)
            cv2.circle(img, pt, 14, (255, 255, 255), 2)
        h = img.shape[0]
        cv2.putText(img, "Click on the path where drone should STOP",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, "ENTER = confirm   Q = abort",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        displays[0] = img

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if path_points:
            px = np.array([p[0] for p in path_points], dtype=np.float32)
            py = np.array([p[1] for p in path_points], dtype=np.float32)
            idx = int(np.argmin(np.abs(px - x)))   # snap to same x on path
            selected[0] = (int(px[idx]), int(py[idx]))
        else:
            selected[0] = (x, y)
        _redraw(selected[0])

    _redraw()
    cv2.namedWindow("Select Stop Point")
    cv2.setMouseCallback("Select Stop Point", on_mouse)

    while True:
        cv2.imshow("Select Stop Point", displays[0])
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and selected[0] is not None:   # ENTER
            break
        if key in (ord('q'), ord('Q')):
            selected[0] = None
            break

    cv2.destroyWindow("Select Stop Point")
    return selected[0]


# ── Flight map ────────────────────────────────────────────────────────────────

def _draw_flight_map(path_points, trail, drone_center, target,
                     frame_w: int, frame_h: int) -> np.ndarray:
    """
    Schematic side-view map shown in a separate window during flight.
    Coordinates are taken directly from the camera frame so the map
    matches what the camera sees (x = rail direction, y = height).
    """
    MAP_W, MAP_H = 900, 300
    PAD = 50

    canvas = np.full((MAP_H, MAP_W, 3), 28, dtype=np.uint8)  # dark background

    # Grid lines
    for gx in range(PAD, MAP_W - PAD, (MAP_W - 2 * PAD) // 8):
        cv2.line(canvas, (gx, PAD), (gx, MAP_H - PAD), (50, 50, 50), 1)
    for gy in range(PAD, MAP_H - PAD, (MAP_H - 2 * PAD) // 4):
        cv2.line(canvas, (PAD, gy), (MAP_W - PAD, gy), (50, 50, 50), 1)

    def to_map(x, y):
        mx = PAD + int((x / max(frame_w, 1)) * (MAP_W - 2 * PAD))
        my = PAD + int((y / max(frame_h, 1)) * (MAP_H - 2 * PAD))
        return (_clamp(mx, 0, MAP_W - 1), _clamp(my, 0, MAP_H - 1))

    # Rail path — red
    if path_points and len(path_points) > 1:
        pts = np.array([to_map(p[0], p[1]) for p in path_points], dtype=np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (60, 60, 220), 2)

    # Drone trail — fading yellow→green
    if len(trail) > 1:
        mapped = [to_map(p[0], p[1]) for p in trail]
        n = len(mapped)
        for i in range(1, n):
            t     = i / n
            color = (0, int(180 + 75 * t), int(255 * t))
            cv2.line(canvas, mapped[i - 1], mapped[i], color, 2)

    # Landing target — cyan
    if target is not None:
        tm = to_map(target[0], target[1])
        cv2.circle(canvas, tm, 14, (0, 220, 220), -1)
        cv2.circle(canvas, tm, 14, (255, 255, 255), 2)
        cv2.putText(canvas, "LAND", (tm[0] + 16, tm[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)

    # Drone current position — bright green
    if drone_center is not None:
        dm = to_map(drone_center[0], drone_center[1])
        cv2.circle(canvas, dm, 11, (0, 255, 80), -1)
        cv2.circle(canvas, dm, 11, (255, 255, 255), 2)
        cv2.putText(canvas, "DRONE", (dm[0] + 13, dm[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)

    # Labels
    cv2.putText(canvas, "← left", (PAD, MAP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(canvas, "right →", (MAP_W - PAD - 55, MAP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(canvas, "high", (4, PAD + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(canvas, "low", (4, MAP_H - PAD - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(canvas, "FLIGHT MAP", (MAP_W // 2 - 55, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    return canvas


# ── Trail drawing ─────────────────────────────────────────────────────────────

def _draw_trail(frame: np.ndarray, trail: list[tuple[int, int]]) -> np.ndarray:
    """
    Draw the drone's flight history as a fading yellow-to-orange polyline.
    Recent positions are bright yellow; older ones fade toward orange/dim.
    """
    if len(trail) < 2:
        return frame
    out = frame.copy()
    n   = len(trail)
    for i in range(1, n):
        t     = i / n                          # 0 = oldest, 1 = newest
        color = (0, int(180 + 75 * t), int(255 * t))   # dim→bright yellow
        thickness = 2 if t < 0.7 else 3
        cv2.line(out, trail[i - 1], trail[i], color, thickness)
    # Dot at current position
    cv2.circle(out, trail[-1], 5, (0, 255, 255), -1)
    return out


# ── HUD helpers ───────────────────────────────────────────────────────────────

def _preflight_hud(frame, smoother, drone_detected):
    out  = frame.copy()
    h, w = out.shape[:2]

    path_ok = smoother.is_stable
    score   = smoother.stability_score
    count   = smoother._frame_count
    can_arm = path_ok and drone_detected

    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.5, overlay, 0.5, 0)

    def _row(txt, color, row):
        cv2.putText(out, txt, (14, h - 100 + 22 + row * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    if path_ok:
        _row("PATH STABLE", (0, 220, 0), 0)
    else:
        needed = max(0, smoother.MIN_FRAMES - count)
        _row(f"WARMING UP ({needed} frames left)" if needed > 0
             else f"PATH UNSTABLE  score={score:.1f}px", (0, 60, 255), 0)

    _row("DRONE DETECTED" if drone_detected else "DRONE NOT DETECTED",
         (0, 220, 0) if drone_detected else (0, 60, 255), 1)
    _row("Waiting for both conditions... Q = quit",
         (0, 220, 0) if can_arm else (180, 180, 180), 2)

    bar_w  = w - 28
    filled = int(bar_w * max(0.0, 1.0 - min(score / 15.0, 1.0)))
    cv2.rectangle(out, (14, h - 10), (14 + bar_w, h - 3), (60, 60, 60), -1)
    cv2.rectangle(out, (14, h - 10), (14 + filled, h - 3),
                  (0, 220, 0) if path_ok else (0, 60, 255), -1)
    return out


def _flight_hud(frame, pitch, throttle, dist, remaining, height_cm, target,
                drone_center=None, total_dist_px=None):
    out = frame.copy()
    dist_s   = f"{dist:.0f}px" if dist is not None else "--"
    height_s = f"{height_cm:.0f}cm" if height_cm is not None else "--"

    # Distance remaining to target along x axis
    if drone_center is not None and target is not None:
        px_remaining = max(0, target[0] - drone_center[0])
        if total_dist_px and total_dist_px > 0:
            pct = 100.0 * (1.0 - px_remaining / total_dist_px)
            to_target_s = f"{px_remaining}px  ({pct:.0f}% done)"
        else:
            to_target_s = f"{px_remaining}px"
    else:
        to_target_s = "--"

    lines = [
        (f"pitch:        {pitch:+d}%",          (255, 255, 255)),
        (f"throttle:     {throttle:+d}%",       (255, 255, 255)),
        (f"dist to path: {dist_s}",             (0, 220, 255)),
        (f"to target:    {to_target_s}",        (0, 255, 128)),
        (f"height:       {height_s}",           (0, 220, 255)),
        (f"time left:    {remaining:.1f}s",     (255, 255, 255)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(out, txt, (10, 25 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    # Draw target marker
    if target is not None:
        cv2.circle(out, target, 14, (0, 255, 255), -1)
        cv2.circle(out, target, 14, (255, 255, 255), 2)
        cv2.putText(out, "TARGET", (target[0] + 16, target[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    device, drone_model, segment_model = railDetection.load_models()
    grabber  = railDetection.FrameGrabber(CAMERA_SERVER)
    smoother = railDetection.PathSmoother(alpha=0.2)

    if not grabber.isOpened():
        raise RuntimeError("Could not open camera stream")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 1 — Pre-flight: both models, wait for stable path + drone visible
    # ════════════════════════════════════════════════════════════════════════
    print("[PHASE 1] Waiting for stable path and drone detection...")
    frozen_frame = None

    while True:
        ret, frame = grabber.read()
        if not ret or frame is None:
            continue

        result = railDetection.process_frame(
            frame, drone_model, segment_model, device, smoother,
            detect_only=False)

        drone_detected = result.drone_center is not None
        can_arm        = smoother.is_stable and drone_detected

        display = _preflight_hud(result.annotated_frame, smoother, drone_detected)
        cv2.imshow("Drone Flight", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("[QUIT]")
            grabber.release()
            cv2.destroyAllWindows()
            return

        if can_arm:
            smoother.lock()
            frozen_frame       = result.annotated_frame.copy()
            frozen_drone_center = result.drone_center
            print("[PHASE 1 DONE] Path locked, drone detected.")
            break

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2 — Target selection: freeze frame, user clicks stop point
    # ════════════════════════════════════════════════════════════════════════
    print("[PHASE 2] Select stop point on the frozen frame...")
    cv2.destroyWindow("Drone Flight")

    target = _select_target(frozen_frame, smoother.update(None, frozen_frame.shape[1]))

    if target is None:
        print("[ABORT] No target selected.")
        grabber.release()
        cv2.destroyAllWindows()
        return

    target_x       = target[0]
    total_dist_px  = max(1, target_x - (frozen_drone_center[0] if frozen_drone_center else 0))
    print(f"[PHASE 2 DONE] Target set at pixel x={target_x}  "
          f"(~{total_dist_px}px from drone start).")

    # ── Save flight plan snapshot ─────────────────────────────────────────────
    snapshot = frozen_frame.copy()
    # Target marker (cyan)
    cv2.circle(snapshot, target, 18, (0, 255, 255), -1)
    cv2.circle(snapshot, target, 18, (255, 255, 255), 3)
    cv2.putText(snapshot, "LANDING", (target[0] + 20, target[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Drone start marker (green), if we have it
    if frozen_drone_center is not None:
        cv2.circle(snapshot, frozen_drone_center, 14, (0, 255, 0), -1)
        cv2.circle(snapshot, frozen_drone_center, 14, (255, 255, 255), 2)
        cv2.putText(snapshot, "DRONE START",
                    (frozen_drone_center[0] + 16, frozen_drone_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        # Distance label along the bottom
        cv2.putText(snapshot, f"distance: ~{total_dist_px}px",
                    (10, snapshot.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(SNAPSHOT_PATH, snapshot)
    print(f"[SNAPSHOT] Saved flight plan → {SNAPSHOT_PATH}")

    # ════════════════════════════════════════════════════════════════════════
    # Phase 3 — Flight: detection only, follow path to target
    # ════════════════════════════════════════════════════════════════════════
    drone = Drone()
    drone.connect()

    try:
        print("[TAKEOFF]")
        drone.takeoff()
        time.sleep(2.0)

        # Determine travel direction from drone start → target.
        # Positive pitch moves drone in +x in the frame; negate if target is to the left.
        start_x   = frozen_drone_center[0] if frozen_drone_center else target_x
        direction = 1 if target_x >= start_x else -1
        pitch_cmd = direction * FORWARD_PITCH
        print(f"[PHASE 3] Flying to target x={target_x}  "
              f"({'right' if direction > 0 else 'left'} in frame, pitch={pitch_cmd:+d}%)...")
        flight_start    = time.time()
        last_cmd_time   = 0.0
        trail: list[tuple[int, int]] = []

        while True:
            ret, frame = grabber.read()
            if not ret or frame is None:
                continue

            # Detection only — segmentation model no longer runs
            result    = railDetection.process_frame(
                frame, drone_model, segment_model, device, smoother,
                detect_only=True)

            elapsed   = time.time() - flight_start
            remaining = max(0.0, FLIGHT_DURATION_SEC - elapsed)
            height_cm = _get_height_cm(drone)

            # ── Stop conditions ────────────────────────────────────────────
            # if elapsed >= FLIGHT_DURATION_SEC:
            #     print("[TIMEOUT] Safety timeout reached.")
            #     break

            if height_cm is not None and height_cm > MAX_HEIGHT_CM:
                print(f"[EMERGENCY] Height {height_cm:.0f}cm > {MAX_HEIGHT_CM}cm — landing!")
                break

            if result.pad_detected:
                print("[PAD] Landing pad reached.")
                break

            if result.drone_center is not None:
                dx = (result.drone_center[0] - target_x) * direction
                if dx >= -TARGET_REACH_PX:
                    print(f"[TARGET] Reached x={result.drone_center[0]} — landing.")
                    break

            # ── Accumulate trail ──────────────────────────────────────────
            if result.drone_center is not None:
                trail.append(result.drone_center)

            # ── Commands ──────────────────────────────────────────────────
            if result.drone_center is not None:
                pitch, throttle, dist = _compute_controls(
                    result.drone_center, result.path_points, pitch_cmd)
            else:
                # Drone not visible — keep moving, hold altitude
                pitch, throttle, dist = pitch_cmd, BASE_THROTTLE, None

            now = time.time()
            if now - last_cmd_time >= COMMAND_INTERVAL_SEC:
                drone.set_pitch(pitch)
                drone.set_throttle(throttle)
                drone.set_roll(0)
                drone.set_yaw(0)
                drone.move()
                last_cmd_time = now

            # ── Display ───────────────────────────────────────────────────
            annotated = _flight_hud(result.annotated_frame,
                                    pitch, throttle, dist,
                                    remaining, height_cm, target,
                                    drone_center=result.drone_center,
                                    total_dist_px=total_dist_px)
            annotated = _draw_trail(annotated, trail)
            cv2.imshow("Drone Flight", annotated)

            fh, fw = frame.shape[:2]
            flight_map = _draw_flight_map(
                result.path_points, trail, result.drone_center, target, fw, fh)
            cv2.imshow("Flight Map", flight_map)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("[ABORT]")
                break
            if key in (ord('l'), ord('L')):
                print("[LAND] Manual land triggered.")
                break

    finally:
        drone.land()
        time.sleep(1)
        drone.disconnect()
        grabber.release()
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    run()
