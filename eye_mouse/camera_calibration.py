"""Optional measured camera optics using a printed chessboard and diverse views."""

import time

import cv2
import numpy as np

from .storage import atomic_json, utc_now


def calibrate_camera(args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    pattern = (args.columns, args.rows)
    objects = np.zeros((args.columns * args.rows, 3), np.float32)
    objects[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2) * args.square_mm
    world, pixels = [], []
    previous = None
    size = None
    print("Show a PRINTED chessboard. SPACE captures a view; ENTER solves (12+ views); ESC cancels.")
    print("Vary board tilt, distance and position across the whole image. Keep camera focus fixed.")
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera; stop the lab camera before this utility")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCornersSB(gray, pattern)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern, corners, True)
            cv2.putText(
                display,
                f"Views: {len(pixels)} | SPACE capture | ENTER solve | ESC cancel",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 230, 230),
                2,
            )
            cv2.imshow("Camera intrinsics calibration", display)
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                return
            if key == 32 and found:
                if previous is not None and np.mean(np.linalg.norm(corners - previous, axis=2)) < 12:
                    print("View too similar: move/tilt the board before capturing again.")
                    continue
                world.append(objects.copy())
                pixels.append(corners)
                previous = corners.copy()
                print(f"Captured view {len(pixels)}")
            if key in (10, 13):
                if len(pixels) < 12:
                    print("Capture at least 12 diverse views first.")
                    continue
                rms, k, distortion, rotations, translations = cv2.calibrateCamera(
                    world, pixels, size, None, None
                )
                errors = []
                for obj, img, r, t in zip(world, pixels, rotations, translations):
                    projected, _ = cv2.projectPoints(obj, r, t, k, distortion)
                    errors.append(float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2)))))
                if not np.isfinite(k).all() or not np.isfinite(distortion).all():
                    raise RuntimeError("Calibration produced non-finite intrinsics; collect better views")
                atomic_json(
                    args.output,
                    {
                        "schema_version": 1,
                        "created_utc": utc_now(),
                        "camera": args.camera,
                        "image_size": list(size),
                        "camera_matrix": k.tolist(),
                        "distortion": distortion.tolist(),
                        "reprojection_rms_px": rms,
                        "per_view_rms_px": errors,
                        "views": len(world),
                        "board_inner_corners": list(pattern),
                        "square_mm": args.square_mm,
                    },
                )
                print(
                    f"Saved {args.output}; reprojection RMS {rms:.3f} px. This is camera calibration error, not gaze error."
                )
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.05)
