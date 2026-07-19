import cv2

import railDetection

# Phone and laptop must share same Wi-Fi network.
CAMERA_SERVER = "https://192.168.1.133:8080/video"


if __name__ == "__main__":
    device, drone_model, segment_model = railDetection.load_models()

    grabber = railDetection.FrameGrabber(CAMERA_SERVER)
    smoother = railDetection.PathSmoother(alpha=0.2)

    if not grabber.isOpened():
        print("Could not open stream")
        exit()

    path_locked = False

    while True:
        ret, frame = grabber.read()
        if not ret or frame is None:
            continue

        # Lock the moment the path converges — segmentation never runs again after this.
        if not path_locked and smoother.is_stable:
            smoother.lock()
            path_locked = True
            print("[PATH LOCKED] Segmentation stopped — path is frozen.")

        result = railDetection.process_frame(
            frame, drone_model, segment_model, device, smoother,
            detect_only=path_locked)

        # HUD overlay
        h, w = result.annotated_frame.shape[:2]
        lock_label = "PATH LOCKED" if path_locked else f"warming up  score={smoother.stability_score:.1f}px"
        lines = [
            lock_label,
            f"drone: {result.drone_center}",
            f"rail_cx: {result.rail_centerline_x}",
            f"lateral_offset: {result.lateral_offset:+d}px" if result.lateral_offset is not None else "lateral_offset: --",
            f"vertical_offset: {result.vertical_offset:+d}px" if result.vertical_offset is not None else "vertical_offset: --",
            f"PAD REACHED" if result.pad_detected else "pad: not reached",
        ]
        for i, line in enumerate(lines):
            color = (0, 220, 0) if (i == 0 and path_locked) else \
                    (0, 0, 255) if (i == 5 and result.pad_detected) else \
                    (255, 255, 255)
            cv2.putText(result.annotated_frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Drone Rail Detection", result.annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabber.release()
    cv2.destroyAllWindows()
