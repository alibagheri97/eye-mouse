"""Image-space measurements; these are not an eyeball or gaze-ray reconstruction."""

import math
from dataclasses import dataclass, field

import numpy as np

FEATURE_NAMES = [
    "right_iris_x",
    "right_iris_y",
    "left_iris_x",
    "left_iris_y",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
    "face_x",
    "face_y",
    "log_face_scale",
]


@dataclass
class Observation:
    timestamp: float
    features: np.ndarray | None = None
    landmarks: np.ndarray | None = None
    valid: bool = False
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    inference_ms: float = 0.0


def eye_geometry(points, corners, lids, iris):
    a, b = points[list(corners), :2]
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1:
        raise ValueError("Eye too small")
    horizontal = axis / width
    vertical = np.array([-horizontal[1], horizontal[0]])
    center = points[iris, :2]
    offset = center - (a + b) / 2
    local = np.array([offset @ horizontal / width, offset @ vertical / width])
    opening = abs(float((points[lids[1], :2] - points[lids[0], :2]) @ vertical)) / width
    return local, width, opening


def rotation_angles(matrix):
    # Remove any uniform scaling before extracting Euler angles (XYZ convention).
    rotation = np.array(matrix, dtype=float)[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return yaw, pitch, roll


def measure(points, matrix, blendshapes, image_size, settings, focal_px=None):
    width, height = image_size
    right, rw, ro = eye_geometry(points, (33, 133), (159, 145), 468)
    left, lw, lo = eye_geometry(points, (362, 263), (386, 374), 473)
    yaw, pitch, roll = rotation_angles(matrix)
    pupils = float(np.linalg.norm(points[468, :2] - points[473, :2]))
    scale = float(np.linalg.norm(points[234, :2] - points[454, :2])) / width
    if scale <= 0 or pupils < 1:
        raise ValueError("Degenerate face geometry")
    center = (points[234, :2] + points[454, :2]) / 2 / [width, height]
    features = np.array([*right, *left, yaw, pitch, roll, *center, math.log(scale)])
    focal = focal_px if focal_px else width / (2 * math.tan(math.radians(settings.hfov) / 2))
    distance = focal * settings.ipd_mm * math.cos(yaw) / pupils
    blink = max(blendshapes.get("eyeBlinkLeft", 0.0), blendshapes.get("eyeBlinkRight", 0.0))
    reasons = []
    if min(ro, lo) < settings.min_eye_open or blink > settings.max_blink:
        reasons.append("Eyes closed / blink")
    if min(rw, lw) < settings.min_eye_width_px:
        reasons.append("Eyes too small; move closer or increase camera resolution")
    if max(abs(math.degrees(yaw)), abs(math.degrees(pitch))) > settings.max_pose_degrees:
        reasons.append("Head angle outside allowed range")
    if not np.all(np.isfinite(features)) or np.any(np.abs(np.r_[right, left]) > 0.65):
        reasons.append("Unreliable iris geometry")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] > width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] > height)
    ):
        reasons.append("Face partly outside camera")
    metrics = dict(
        yaw_deg=math.degrees(yaw),
        pitch_deg=math.degrees(pitch),
        roll_deg=math.degrees(roll),
        distance_mm=distance,
        ipd_px=pupils,
        right_eye_px=rw,
        left_eye_px=lw,
        right_open=ro,
        left_open=lo,
        blink=blink,
        focal_px=focal,
        face_scale=scale,
    )
    return features, metrics, reasons
