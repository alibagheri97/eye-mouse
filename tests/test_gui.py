"""Interaction checks with generated observations; no claim of human gaze accuracy."""

import json
import os
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_calibration import calibration_visits

from eye_mouse.experiment import Experiment
from eye_mouse.geometry import Observation
from eye_mouse.gui import LabWindow, TargetWindow
from eye_mouse.settings import Settings


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def lab(app, tmp_path, monkeypatch):
    import eye_mouse.gui as gui
    import eye_mouse.storage as storage

    monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    window = LabWindow()
    window.problem = lambda message: pytest.fail(message)
    window.show()
    app.processEvents()
    yield window
    window.worker = None
    window.close()
    app.processEvents()
    import shiboken6

    if shiboken6.isValid(window):
        shiboken6.delete(window)


def test_initial_controls_and_fullscreen_keyboard_cancel(app, lab, tmp_path):
    assert lab.start_button.isEnabled()
    assert not lab.arm_button.isEnabled()
    lab.camera_info = {"camera": 0, "width": 1280, "height": 720, "model_sha256": "test"}
    lab.latest = Observation(time.monotonic(), np.zeros(10), valid=True)
    lab.start_experiment("calibration")
    target = lab.target_window
    assert isinstance(target, TargetWindow)
    QTest.keyClick(target, Qt.Key.Key_Space)
    app.processEvents()
    assert lab.experiment.onset is not None
    assert target.running
    QTest.keyClick(target, Qt.Key.Key_S)
    app.processEvents()
    assert lab.experiment.index == 1
    QTest.keyClick(target, Qt.Key.Key_Escape)
    app.processEvents()
    assert lab.experiment is None
    assert lab.target_window is None
    assert not lab.cursor.armed
    saved = list((tmp_path / "sessions").glob("*/session.json"))
    assert json.loads(saved[0].read_text())["outcome"] == "cancelled"


def test_calibration_completion_saves_loadable_profile_and_independent_validation(app, lab, tmp_path):
    lab.camera_info = {"camera": 0, "width": 1280, "height": 720, "model_sha256": "test"}
    lab.experiment = Experiment("calibration", Settings())
    lab.experiment.visits = calibration_visits()
    lab.experiment.index = len(lab.experiment.schedule)
    lab.finish_experiment()
    assert lab.model is not None
    profile = json.loads((tmp_path / "profiles" / "latest.json").read_text())
    assert profile["model"]["version"] == 1
    assert not lab.cursor.armed
    e = Experiment("validation", Settings())
    from test_calibration import features

    from eye_mouse.calibration import Visit

    e.visits = [
        Visit(list(target), group, [features(*target).tolist()] * 24, list(range(24)), 26, 2)
        for target, group in e.schedule
    ]
    e.index = len(e.schedule)
    lab.experiment = e
    lab.finish_experiment()
    reports = list((tmp_path / "sessions").glob("*-validation/session.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())["report"]
    assert report["completed_targets"] == 16
    assert (reports[0].parent / "validation.png").exists()


def test_close_with_active_acquisition_closes_target_window(app, lab):
    lab.camera_info = {"camera": 0, "width": 1280, "height": 720}
    lab.latest = Observation(time.monotonic(), np.zeros(10), valid=True)
    lab.start_experiment("validation")
    target = lab.target_window
    lab.close()
    app.processEvents()
    import shiboken6

    assert not shiboken6.isValid(target) or not target.isVisible()
    assert lab.experiment is None
