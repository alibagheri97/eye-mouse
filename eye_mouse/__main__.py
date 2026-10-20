import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Local experimental gaze/mouse laboratory")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("download-model", help="Download official MediaPipe face landmark model")
    doctor = commands.add_parser("doctor", help="Check display/model and optionally exercise a real camera")
    doctor.add_argument("--camera", type=int, default=None, help="Open camera and run local inference")
    doctor.add_argument("--frames", type=int, default=30)
    doctor.add_argument("--output", default="", help="Optional JSON diagnostic output")
    intrinsics = commands.add_parser(
        "camera-calibrate", help="Measure camera intrinsics with a printed chessboard"
    )
    intrinsics.add_argument("--camera", type=int, default=0)
    intrinsics.add_argument("--width", type=int, default=1280)
    intrinsics.add_argument("--height", type=int, default=720)
    intrinsics.add_argument("--columns", type=int, default=9, help="INNER corners horizontally, not squares")
    intrinsics.add_argument("--rows", type=int, default=6, help="INNER corners vertically")
    intrinsics.add_argument("--square-mm", type=float, default=25.0)
    intrinsics.add_argument("--output", default="data/camera-intrinsics.json")
    parser.add_argument("--smoke-seconds", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from .settings import MODEL_PATH, Settings

    if args.command == "download-model":
        from .tracker import ensure_model

        path = ensure_model()
        print(f"Model: {path}\nSHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
        return
    if args.command == "doctor":
        import cv2
        import mediapipe as mp

        from .tracker import Tracker

        report = {
            "model_present": MODEL_PATH.exists(),
            "model_path": str(MODEL_PATH),
            "display": os.environ.get("DISPLAY"),
            "session_type": os.environ.get("XDG_SESSION_TYPE"),
            "mediapipe": mp.__version__,
            "opencv": cv2.__version__,
        }
        if args.camera is not None:
            cap = cv2.VideoCapture(args.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            tracker = None
            try:
                if not cap.isOpened():
                    raise RuntimeError("Cannot open camera")
                tracker = Tracker(Settings(camera=args.camera))
                observations = []
                start = time.monotonic()
                for _ in range(max(1, args.frames)):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera read failed")
                    _, obs = tracker.process(frame)
                    observations.append(obs)
                report.update(
                    camera=args.camera,
                    capture_size=[frame.shape[1], frame.shape[0]],
                    frames=len(observations),
                    faces=sum(o.landmarks is not None for o in observations),
                    valid_frames=sum(o.valid for o in observations),
                    last_reasons=observations[-1].reasons,
                    last_metrics=observations[-1].metrics,
                    elapsed_seconds=time.monotonic() - start,
                    mean_inference_ms=sum(o.inference_ms for o in observations) / len(observations),
                )
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                cap.release()
                if tracker:
                    tracker.close()
        print(json.dumps(report, indent=2))
        if args.output:
            from .storage import atomic_json

            atomic_json(args.output, report)
        if "error" in report or not report["model_present"]:
            raise SystemExit(1)
        return
    if args.command == "camera-calibrate":
        if args.columns < 3 or args.rows < 3 or args.square_mm <= 0:
            parser.error("Chessboard needs at least 3 × 3 inner corners and positive square size")
        from .camera_calibration import calibrate_camera

        calibrate_camera(args)
        return
    # OpenCV wheels bundle Qt plugins for cv2.imshow; do not let those override PySide6's Qt.
    import cv2  # noqa: F401
    from PySide6.QtCore import QLibraryInfo, Qt, QTimer

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
        Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms"
    )
    os.environ.pop("QT_QPA_FONTDIR", None)
    from PySide6.QtWidgets import QApplication

    from .gui import LabWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Eye Mouse Lab")
    window = LabWindow()
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.show()
    if args.smoke_seconds:
        QTimer.singleShot(int(args.smoke_seconds * 1000), window.close)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
