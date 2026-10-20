import json
import math

import numpy as np
import pytest

from eye_mouse.calibration import Visit
from eye_mouse.filtering import OneEuro
from eye_mouse.geometry import eye_geometry, rotation_angles
from eye_mouse.settings import Settings
from eye_mouse.storage import atomic_json, compatibility, save_session


def test_eye_coordinates_are_invariant_to_translation_scale_and_roll():
    points = np.array([[10, 10], [50, 10], [30, 4], [30, 16], [35, 12]], dtype=float)
    local, width, opening = eye_geometry(points, (0, 1), (2, 3), 4)
    angle = 0.7
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    transformed = (points @ rotation.T) * 3 + [100, 70]
    again, scaled_width, scaled_open = eye_geometry(transformed, (0, 1), (2, 3), 4)
    np.testing.assert_allclose(local, again, atol=1e-12)
    assert scaled_width == pytest.approx(width * 3)
    assert scaled_open == pytest.approx(opening)


def test_rotation_zero_and_known_yaw():
    np.testing.assert_allclose(rotation_angles(np.eye(4)), [0, 0, 0])
    angle = 0.2
    rotation = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )
    yaw, pitch, roll = rotation_angles(rotation)
    assert yaw == pytest.approx(angle)
    assert pitch == pytest.approx(0)
    assert roll == pytest.approx(0)


def test_filter_reduces_stationary_noise_and_resets_after_loss():
    rng = np.random.default_rng(1)
    noisy = rng.normal(0, 12, (300, 2))
    filt = OneEuro(min_cutoff=1.2, beta=0)
    smoothed = np.array([filt.update(v, i / 30) for i, v in enumerate(noisy)])
    assert smoothed[50:].std() < noisy.std() * 0.6
    np.testing.assert_allclose(filt.update([1200, 800], 12), [1200, 800])


def test_atomic_storage_and_complete_session_export(tmp_path, monkeypatch):
    import eye_mouse.storage as storage

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    path = tmp_path / "profile.json"
    atomic_json(path, {"value": 1})
    atomic_json(path, {"value": 2})
    assert json.loads(path.read_text())["value"] == 2
    assert not path.with_suffix(".json.tmp").exists()
    visit = Visit([0.5, 0.5], 0, [[0] * 10], [10.0], 3, 2)
    folder = save_session("validation", [visit], Settings(), {"test": True}, trace=[{"valid": False}])
    report = json.loads((folder / "session.json").read_text())
    assert report["acquisition_trace"] == [{"valid": False}]
    assert report["visits"][0]["samples"] == [[0] * 10]
    assert len((folder / "samples.csv").read_text().splitlines()) == 2


def test_compatibility_catches_optics_resolution_and_screen_changes(tmp_path):
    settings = Settings()
    camera = {"width": 1280, "height": 720, "model_sha256": "abc"}
    screen = {"name": "display", "geometry": [0, 0, 1920, 1080], "dpr": 1}
    original = compatibility(settings, camera, screen)
    assert original != compatibility(settings, {**camera, "width": 640}, screen)
    settings.hfov = 70
    assert original != compatibility(settings, camera, screen)
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text("{}")
    settings.intrinsics_path = str(intrinsics)
    first = compatibility(settings, camera, screen)
    intrinsics.write_text('{"changed":true}')
    assert first != compatibility(settings, camera, screen)
