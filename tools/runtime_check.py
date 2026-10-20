"""Explicit desktop smoke check. It saves no camera images and does not train human gaze."""

import argparse
import json
import os
import time
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, Qt, QTimer

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
    Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / "platforms"
)
os.environ.pop("QT_QPA_FONTDIR", None)

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from eye_mouse.gui import LabWindow
from eye_mouse.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--output", default="artifacts/gui-runtime.json")
    args = parser.parse_args()
    report = {"camera_requested": args.camera, "events": []}
    app = QApplication([])
    lab = LabWindow()
    lab.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    lab.problem = lambda message: report["events"].append({"error": message})
    lab.show()
    started = time.monotonic()

    def inspect():
        report["window_visible"] = lab.isVisible()
        report["screen"] = lab.screen_info()
        report["camera_info"] = lab.camera_info
        report["camera_frames_observed"] = len(lab.frame_times)
        report["diagnostics"] = lab.diagnostics.text()
        if args.camera and lab.latest and lab.latest.valid:
            lab.start_experiment("calibration")
            QTest.keyClick(lab.target_window, Qt.Key.Key_Space)
            report["target_window_opened"] = lab.target_window.isVisible()
        else:
            report["target_window_opened"] = False

    def cancel_and_stop():
        if lab.target_window:
            report["target_acquisition_attempts"] = lab.experiment.attempted
            QTest.keyClick(lab.target_window, Qt.Key.Key_Escape)
            report["escape_cancelled_acquisition"] = lab.experiment is None
        lab.stop_camera()

    def finish():
        if lab.worker is not None and time.monotonic() - started < 20:
            QTimer.singleShot(100, finish)
            return
        report["camera_released"] = lab.worker is None
        report["mouse_disarmed"] = not lab.cursor.armed
        report["elapsed_seconds"] = time.monotonic() - started
        # Screenshot only after camera pixels have been cleared from every preview.
        if lab.worker is None:
            destination = Path(args.output).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            lab.grab().save(str(destination))
            report["stopped_window_screenshot"] = str(destination)
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2))
        lab.close()

    if args.camera:
        QTimer.singleShot(100, lab.start_camera)
    QTimer.singleShot(4500 if args.camera else 100, inspect)
    QTimer.singleShot(7500 if args.camera else 400, cancel_and_stop)
    QTimer.singleShot(9500 if args.camera else 1000, finish)
    app.exec()
    if report.get("events") or not report.get("camera_released"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
