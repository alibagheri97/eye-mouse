import os
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("EYE_MOUSE_DATA", "data")).resolve()
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@dataclass
class Settings:
    camera: int = 0
    width: int = 1280
    height: int = 720
    camera_fps: int = 30
    hfov: float = 60.0
    ipd_mm: float = 63.0
    screen_width_mm: float = 310.0
    viewing_distance_mm: float = 600.0
    min_eye_open: float = 0.16
    max_blink: float = 0.45
    min_eye_width_px: float = 22.0
    max_pose_degrees: float = 30.0
    settle_seconds: float = 0.8
    sample_seconds: float = 1.2
    min_samples: int = 18
    target_timeout: float = 15.0
    fixation_jitter: float = 0.045
    calibration_grid: int = 5
    calibration_passes: int = 2
    min_cutoff: float = 1.2
    filter_beta: float = 0.015
    intrinsics_path: str = ""

    def to_dict(self):
        return asdict(self)
