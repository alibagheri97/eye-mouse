import csv
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from .geometry import FEATURE_NAMES
from .settings import DATA_DIR


def utc_now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def compatibility(settings, camera_info, screen):
    intrinsics_hash = ""
    if settings.intrinsics_path:
        intrinsics_hash = hashlib.sha256(Path(settings.intrinsics_path).read_bytes()).hexdigest()
    return {
        "camera": settings.camera,
        "capture_size": [camera_info["width"], camera_info["height"]],
        "screen": screen,
        "intrinsics_sha256": intrinsics_hash,
        "hfov": settings.hfov,
        "ipd_mm": settings.ipd_mm,
        "model_sha256": camera_info.get("model_sha256", ""),
    }


def save_session(kind, visits, settings, metadata, report=None, model=None, outcome="completed", trace=None):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder = DATA_DIR / "sessions" / f"{stamp}-{kind}"
    folder.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "kind": kind,
        "outcome": outcome,
        "settings": settings.to_dict(),
        "metadata": metadata,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("mediapipe", "opencv-contrib-python", "numpy", "PySide6")
            },
        },
        "feature_names": FEATURE_NAMES,
        "visits": [v.to_dict() for v in visits],
        "report": report,
        "model": model.to_dict() if model else None,
        "acquisition_trace": trace or [],
    }
    atomic_json(folder / "session.json", payload)
    with (folder / "samples.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["visit", "target_x", "target_y", "status", "monotonic_seconds", *FEATURE_NAMES])
        for i, visit in enumerate(visits):
            for stamp, sample in zip(visit.timestamps, visit.samples):
                writer.writerow([i, *visit.target, visit.status, stamp, *sample])
    return folder
