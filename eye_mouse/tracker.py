import hashlib
import json
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QThread, Signal

from .geometry import Observation, measure
from .settings import MODEL_PATH

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def ensure_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            # A task is a zip bundle. Fail explicitly on HTML/errors/truncated downloads.
            import zipfile

            if not zipfile.is_zipfile(temporary):
                raise ValueError("Downloaded model is not a valid MediaPipe task bundle")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError("Downloaded model checksum differs from the pinned model")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Model checksum mismatch; remove the file and download again")
    return path


class Tracker:
    def __init__(self, settings, model_path=MODEL_PATH):
        self.settings = settings
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1
        self.intrinsics = None
        if settings.intrinsics_path:
            self.intrinsics = json.loads(Path(settings.intrinsics_path).read_text())
            k = np.array(self.intrinsics["camera_matrix"], dtype=float)
            d = np.array(self.intrinsics["distortion"], dtype=float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(d).all():
                raise ValueError("Invalid camera intrinsics")
        self.maps = None

    def process(self, frame, timestamp=None):
        timestamp = time.monotonic() if timestamp is None else timestamp
        start = time.perf_counter()
        h, w = frame.shape[:2]
        focal = None
        if self.intrinsics:
            if [w, h] != self.intrinsics["image_size"]:
                raise ValueError("Intrinsics resolution differs from camera resolution. Recalibrate camera.")
            k = np.array(self.intrinsics["camera_matrix"], dtype=float)
            if self.maps is None:
                self.maps = cv2.initUndistortRectifyMap(
                    k, np.array(self.intrinsics["distortion"]), None, k, (w, h), cv2.CV_32FC1
                )
            frame = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)
            focal = float(k[0, 0])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms = max(self.last_ms + 1, int(timestamp * 1000))
        result = self.landmarker.detect_for_video(image, ms)
        self.last_ms = ms
        obs = Observation(timestamp=timestamp)
        if (
            result.face_landmarks
            and len(result.face_landmarks[0]) >= 478
            and result.facial_transformation_matrixes
        ):
            points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
            blend = (
                {b.category_name: b.score for b in result.face_blendshapes[0]}
                if result.face_blendshapes
                else {}
            )
            obs.landmarks = points
            try:
                obs.features, obs.metrics, obs.reasons = measure(
                    points, result.facial_transformation_matrixes[0], blend, (w, h), self.settings, focal
                )
                obs.valid = not obs.reasons
            except (ValueError, np.linalg.LinAlgError) as exc:
                obs.reasons = [str(exc)]
        else:
            obs.reasons = ["No complete face / iris landmarks"]
        obs.inference_ms = (time.perf_counter() - start) * 1000
        return frame, obs

    def close(self):
        self.landmarker.close()


class CameraWorker(QThread):
    frame_ready = Signal(object, object)
    opened = Signal(object)
    failed = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        cap = tracker = None
        try:
            cap = cv2.VideoCapture(self.settings.camera)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
            cap.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.settings.camera}")
            tracker = Tracker(self.settings)
            self.opened.emit(
                {
                    "camera": self.settings.camera,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "reported_fps": cap.get(cv2.CAP_PROP_FPS),
                    "backend": cap.getBackendName(),
                    "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                }
            )
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                acquired = time.monotonic()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                frame, obs = tracker.process(frame, acquired)
                self.frame_ready.emit(frame, obs)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if tracker:
                tracker.close()
            if cap:
                cap.release()
