"""Plain desktop controls for acquisition, calibration, validation and pointer experiments."""

import json
import time
from collections import deque
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .calibration import GazeModel, validation_report
from .cursor import NativeCursor
from .experiment import Experiment
from .filtering import OneEuro
from .settings import DATA_DIR, MODEL_PATH, Settings
from .storage import atomic_json, compatibility, save_session, utc_now
from .tracker import CameraWorker


class TargetWindow(QWidget):
    cancelled = Signal()
    started = Signal()
    skipped = Signal()
    painted = Signal(float)

    def __init__(self, screen, kind, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.kind = kind
        self.running = False
        self.finishing = False
        self.target = (0.5, 0.5)
        self.gaze = None
        self.message = ""
        self.count = ""
        self.progress = 0.0
        self.show()
        self.activateWindow()
        self.setFocus()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(42, 42, 42))
        p.setPen(QColor(230, 230, 230))
        font = p.font()
        font.setPointSize(13)
        p.setFont(font)
        if not self.running:
            instructions = (
                f"{self.kind.upper()}\n\n"
                "Look at the center of each dot, not the mouse pointer.\n"
                "Keep your head comfortable and steady during each fixation. Blink between dots.\n"
                "Calibration repeats the screen: on the second pass, a small natural posture shift is useful.\n"
                "Keep camera, monitor and chair position unchanged after calibration.\n\n"
                "SPACE: start    ESC: cancel    S: skip a difficult target\n\n"
                "Mouse movement is disabled during acquisition."
            )
            p.drawText(self.rect().adjusted(60, 40, -60, -40), Qt.AlignmentFlag.AlignCenter, instructions)
        else:
            # Keep instructions away from edge targets and the central fixation.
            p.drawText(20, 28, f"{self.kind.upper()}   {self.count}   ESC cancel · S skip")
            p.drawText(20, self.height() - 18, self.message)
            x, y = self.target[0] * self.width(), self.target[1] * self.height()
            p.setPen(QPen(QColor(220, 220, 220), 2))
            p.setBrush(QColor(120, 180, 210))
            p.drawEllipse(int(x - 12), int(y - 12), 24, 24)
            p.setBrush(QColor(15, 15, 15))
            p.drawEllipse(int(x - 3), int(y - 3), 6, 6)
            p.setPen(QPen(QColor(120, 220, 140), 3))
            p.drawArc(int(x - 20), int(y - 20), 40, 40, 90 * 16, -int(360 * 16 * self.progress))
        p.end()
        if self.running:
            self.painted.emit(time.monotonic())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_Space and not self.running:
            self.running = True
            self.started.emit()
        elif event.key() == Qt.Key.Key_S and self.running:
            self.skipped.emit()

    def closeEvent(self, event):
        if not self.finishing:
            self.cancelled.emit()
        event.accept()


class PreviewWindow(QWidget):
    def __init__(self, screen):
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.point = None
        self.show()

    def paintEvent(self, event):
        if self.point is not None:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(QColor(0, 220, 180), 2))
            x, y = self.point
            p.drawEllipse(int(x - 10), int(y - 10), 20, 20)
            p.drawLine(int(x - 15), int(y), int(x + 15), int(y))
            p.drawLine(int(x), int(y - 15), int(x), int(y + 15))


class LabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Mouse — experimental lab")
        self.resize(1180, 850)
        self.worker = None
        self.model = None
        self.profile_context = None
        self.camera_info = None
        self.experiment = self.target_window = self.preview_window = None
        self.cursor = NativeCursor()
        self.latest = None
        self.last_seen = 0
        self.frame_times = deque(maxlen=120)
        self.inference_times = deque(maxlen=120)
        self.ages = deque(maxlen=120)
        self.last_draw = 0
        self.session_paths = []
        self.settings = self.load_settings()
        self.filter = OneEuro(self.settings.min_cutoff, self.settings.filter_beta)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.watchdog)
        self.timer.start(40)
        self.escape = QShortcut(QKeySequence("Escape"), self)
        self.escape.activated.connect(self.emergency_stop)
        self._controls()

    def load_settings(self):
        try:
            data = json.loads((DATA_DIR / "settings.json").read_text())
            names = {f.name for f in fields(Settings)}
            return Settings(**{k: v for k, v in data.items() if k in names})
        except (OSError, ValueError, TypeError):
            return Settings()

    def _button(self, text, action, layout):
        button = QPushButton(text)
        button.clicked.connect(action)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        instructions = QLabel(
            "1. Choose camera/screen and start camera.  2. Check eyes and mesh.  3. Calibrate.  "
            "4. Validate accuracy.  5. Preview gaze, then arm mouse.\n"
            "ESC stops mouse globally while armed. Moving the physical mouse to the desktop's upper-left also stops it."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        setup = QHBoxLayout()
        setup.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 16)
        self.camera_spin.setValue(self.settings.camera)
        setup.addWidget(self.camera_spin)
        setup.addWidget(QLabel("Monitor"))
        self.screen_combo = QComboBox()
        for screen_index, screen in enumerate(QApplication.screens()):
            g = screen.geometry()
            self.screen_combo.addItem(
                f"{screen.name()} — {g.width()} × {g.height()} (scale {screen.devicePixelRatio():g})",
                screen_index,
            )
        setup.addWidget(self.screen_combo, 1)
        self.start_button = self._button("Start camera", self.start_camera, setup)
        self.stop_button = self._button("Stop camera", self.stop_camera, setup)
        layout.addLayout(setup)
        actions = QHBoxLayout()
        self.cal_button = self._button(
            "Calibrate (50 fixations)", lambda: self.start_experiment("calibration"), actions
        )
        self.val_button = self._button(
            "Validate (16 targets)", lambda: self.start_experiment("validation"), actions
        )
        self.preview_check = QCheckBox("Gaze preview")
        self.preview_check.toggled.connect(self.toggle_preview)
        actions.addWidget(self.preview_check)
        self.arm_button = self._button("Arm mouse", self.toggle_mouse, actions)
        self._button("STOP / ESC", self.emergency_stop, actions)
        layout.addLayout(actions)
        self.status = QLabel("Camera stopped. Mouse disarmed. All processing stays on this computer.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        live = QWidget()
        live_layout = QVBoxLayout(live)
        view_options = QHBoxLayout()
        self.mesh_check = QCheckBox("Face mesh")
        self.mesh_check.setChecked(True)
        self.mirror_check = QCheckBox("Mirror preview")
        self.mirror_check.setChecked(True)
        view_options.addWidget(self.mesh_check)
        view_options.addWidget(self.mirror_check)
        view_options.addWidget(QLabel("Preview mirroring does not change calibration coordinates."))
        live_layout.addLayout(view_options)
        self.video_label = QLabel("Start camera to inspect face and eyes")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(600, 300)
        live_layout.addWidget(self.video_label, 1)
        eyes = QHBoxLayout()
        self.eye_labels = [QLabel("Right eye"), QLabel("Left eye")]
        for eye in self.eye_labels:
            eye.setMinimumSize(180, 72)
            eye.setMaximumHeight(95)
            eye.setAlignment(Qt.AlignmentFlag.AlignCenter)
            eyes.addWidget(eye)
        live_layout.addLayout(eyes)
        self.diagnostics = QLabel("No measurements")
        self.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.diagnostics.setWordWrap(True)
        live_layout.addWidget(self.diagnostics)
        tabs.addTab(live, "Live measurements")
        settings_tab = QScrollArea()
        settings_tab.setWidgetResizable(True)
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        self.inputs = {}
        specifications = [
            (
                "Capture / optics (restart camera after changes)",
                [
                    ("width", "Requested camera width (px)", 320, 3840, 1),
                    ("height", "Requested camera height (px)", 240, 2160, 1),
                    ("camera_fps", "Requested camera FPS", 10, 120, 1),
                    ("hfov", "Approx. horizontal camera FOV (degrees)", 25, 110, 1.0),
                    ("ipd_mm", "Your interpupillary distance (mm)", 45, 80, 0.5),
                    ("screen_width_mm", "Measured active monitor width (mm)", 100, 2000, 1.0),
                    ("viewing_distance_mm", "Measured eye-to-screen distance (mm)", 200, 2000, 10.0),
                ],
            ),
            (
                "Fixation acquisition / quality",
                [
                    ("settle_seconds", "Discard after target onset (seconds)", 0.3, 3, 0.1),
                    ("sample_seconds", "Stable fixation duration (seconds)", 0.6, 4, 0.1),
                    ("min_samples", "Minimum accepted frames per target", 8, 100, 1),
                    ("target_timeout", "Skip target after (seconds)", 5, 60, 1.0),
                    ("min_eye_open", "Minimum eyelid opening / eye width", 0.05, 0.4, 0.01),
                    ("max_blink", "Maximum MediaPipe blink score", 0.1, 0.9, 0.05),
                    ("min_eye_width_px", "Minimum eye width in camera pixels", 10, 100, 1.0),
                    ("max_pose_degrees", "Maximum head yaw/pitch (degrees)", 10, 50, 1.0),
                    ("fixation_jitter", "Max median iris deviation / eye width", 0.005, 0.15, 0.005),
                ],
            ),
            (
                "Pointer filter (screen pixels; takes effect immediately)",
                [
                    ("min_cutoff", "One Euro minimum cutoff (Hz)", 0.1, 10, 0.1),
                    ("filter_beta", "Speed response beta", 0.0, 0.1, 0.001),
                ],
            ),
        ]
        for title, specs in specifications:
            group = QGroupBox(title)
            form = QFormLayout(group)
            for key, label, low, high, step in specs:
                is_int = isinstance(getattr(self.settings, key), int)
                control = QSpinBox() if is_int else QDoubleSpinBox()
                control.setRange(low, high)
                control.setSingleStep(step)
                if not is_int:
                    control.setDecimals(3 if step < 0.01 else 2)
                control.setValue(getattr(self.settings, key))
                form.addRow(label, control)
                self.inputs[key] = control
            settings_layout.addWidget(group)
        intrinsics = QHBoxLayout()
        self.intrinsics_label = QLabel(
            self.settings.intrinsics_path or "No measured intrinsics — approximate FOV used"
        )
        self.intrinsics_label.setWordWrap(True)
        intrinsics.addWidget(self.intrinsics_label, 1)
        self.intrinsics_button = self._button("Load camera intrinsics…", self.choose_intrinsics, intrinsics)
        self.clear_intrinsics_button = self._button("Use FOV estimate", self.clear_intrinsics, intrinsics)
        settings_layout.addLayout(intrinsics)
        caveat = QLabel(
            "Distance ≈ focal length × entered IPD × cos(head yaw) / observed iris-center separation.\n"
            "It is approximate camera-to-face axial depth, not eye-to-screen distance. Enter measured eye-to-screen "
            "distance separately for angular errors. Mesh depth is relative, not a depth sensor.\n"
            "For measured focal length/distortion: stop this camera, then run ‘uv run eye-mouse camera-calibrate --help’.\n"
            "Keep exposure/focus stable, light your face evenly, and use a fixed camera. Glasses and reflections may reduce accuracy."
        )
        caveat.setWordWrap(True)
        settings_layout.addWidget(caveat)
        settings_layout.addStretch()
        settings_tab.setWidget(settings_content)
        tabs.addTab(settings_tab, "Experiment settings")
        results = QWidget()
        results_layout = QVBoxLayout(results)
        profile_actions = QHBoxLayout()
        self.save_profile_button = self._button(
            "Save calibration profile…", self.save_profile, profile_actions
        )
        self.load_profile_button = self._button(
            "Load calibration profile…", self.load_profile, profile_actions
        )
        self._button("Open session data folder", self.open_data, profile_actions)
        results_layout.addLayout(profile_actions)
        self.results = QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setPlainText(
            "No accuracy measurement yet.\n\n"
            "Calibration trains the mapping; validation measures it on new target locations.\n"
            "Validation reports raw, unclipped gaze error and fixation precision.\n"
            "The live pointer uses a separately adjustable smoothing filter.\n\n"
            "Each completed or cancelled run automatically saves JSON + raw feature CSV under data/sessions.\n"
            "No camera images or video are saved. Profiles and samples contain personal geometric measurements."
        )
        results_layout.addWidget(self.results)
        tabs.addTab(results, "Results / profiles")
        self.screen_combo.currentIndexChanged.connect(self.screen_changed)
        for key in ("min_cutoff", "filter_beta"):
            self.inputs[key].valueChanged.connect(self.update_filter)

    def screen(self):
        return QApplication.screens()[self.screen_combo.currentData()]

    def screen_info(self):
        screen = self.screen()
        g = screen.geometry()
        return {
            "name": screen.name(),
            "geometry": [g.x(), g.y(), g.width(), g.height()],
            "dpr": screen.devicePixelRatio(),
        }

    def screen_size(self):
        g = self.screen().geometry()
        dpr = self.screen().devicePixelRatio()
        return [round(g.width() * dpr), round(g.height() * dpr)]

    def collect_settings(self):
        for key, control in self.inputs.items():
            setattr(self.settings, key, control.value())
        self.settings.camera = self.camera_spin.value()
        return Settings(**self.settings.to_dict())

    def _controls(self):
        running = self.worker is not None
        busy = self.experiment is not None
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running and not busy)
        self.camera_spin.setEnabled(not running)
        self.screen_combo.setEnabled(not busy and not self.cursor.armed)
        for key, control in self.inputs.items():
            control.setEnabled(
                not busy
                and (
                    not running
                    or key in ("min_cutoff", "filter_beta", "screen_width_mm", "viewing_distance_mm")
                )
            )
        self.intrinsics_button.setEnabled(not running)
        self.clear_intrinsics_button.setEnabled(not running)
        self.cal_button.setEnabled(running and not busy and self.latest is not None)
        self.val_button.setEnabled(running and not busy and self.model is not None)
        self.arm_button.setEnabled(running and not busy and self.model is not None)
        self.arm_button.setText("Disarm mouse" if self.cursor.armed else "Arm mouse")
        self.preview_check.setEnabled(running and not busy and self.model is not None)
        self.save_profile_button.setEnabled(self.model is not None and not busy)
        self.load_profile_button.setEnabled(running and not busy and self.camera_info is not None)

    def start_camera(self):
        if not MODEL_PATH.exists():
            self.problem("Model not installed. Run: uv run eye-mouse download-model")
            return
        settings = self.collect_settings()
        self.model = self.profile_context = self.latest = self.camera_info = None
        self.frame_times.clear()
        self.inference_times.clear()
        self.ages.clear()
        self.filter.reset()
        try:
            atomic_json(DATA_DIR / "settings.json", settings.to_dict())
        except OSError as exc:
            self.problem(f"Cannot save settings: {exc}")
            return
        self.worker = CameraWorker(settings)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.opened.connect(self.on_opened)
        self.worker.failed.connect(self.camera_failed)
        self.worker.finished.connect(self.camera_finished)
        self.status.setText("Opening camera and loading landmark model…")
        self.worker.start()
        self._controls()

    def on_opened(self, info):
        self.camera_info = info
        self.status.setText(
            f"Camera active: {info['width']} × {info['height']} via {info['backend']}. Mouse disarmed."
        )
        self._controls()

    def stop_camera(self):
        self.emergency_stop()
        if self.worker:
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping camera…")

    def camera_failed(self, message):
        self.emergency_stop()
        self.status.setText(f"Camera error: {message}")
        self.results.appendPlainText(f"\nCamera error: {message}")

    def camera_finished(self):
        worker = self.worker
        self.worker = None
        self.latest = None
        self.emergency_stop()
        self.video_label.setText("Camera stopped")
        self.diagnostics.setText("Camera stopped. No live measurements. Mouse disarmed.")
        for label in self.eye_labels:
            label.setText("Camera stopped")
        if self.status.text() == "Stopping camera…":
            self.status.setText("Camera stopped. Mouse disarmed.")
        if worker:
            worker.deleteLater()
        self._controls()

    def on_frame(self, frame, obs):
        if self.worker is None or self.worker.isInterruptionRequested():
            return
        now = time.monotonic()
        age = now - obs.timestamp
        if age > 0.25 or age < -0.01:
            self.filter.reset()
            return
        self.latest = obs
        self.last_seen = now
        self.frame_times.append(obs.timestamp)
        self.inference_times.append(obs.inference_ms)
        self.ages.append(age * 1000)
        if self.experiment and self.target_window and self.target_window.running:
            if self.experiment.kind == "validation" and obs.valid and not self.model.in_domain(obs.features):
                obs.valid = False
                obs.reasons.append("Head outside calibration envelope; return to calibrated posture")
            advanced = self.experiment.update(obs)
            if advanced and self.experiment.done:
                self.finish_experiment()
            elif self.target_window:
                self.refresh_target()
        raw = filtered = None
        in_domain = self.model is not None and obs.valid and self.model.in_domain(obs.features)
        if in_domain and not self.experiment:
            raw = self.model.predict(obs.features) * self.screen_size()
            if np.isfinite(raw).all():
                filtered = self.filter.update(raw, obs.timestamp)
            else:
                raw = None
        else:
            self.filter.reset()
        if self.preview_window:
            self.preview_window.point = (
                (filtered / self.screen().devicePixelRatio()) if filtered is not None else None
            )
            self.preview_window.update()
        if self.cursor.armed and filtered is not None:
            g = self.screen().geometry()
            # Global X11 coordinates, unit scale checked at arming.
            clipped = np.clip(filtered, [3, 3], [g.width() - 4, g.height() - 4])
            try:
                if self.cursor.emergency_requested():
                    self.emergency_stop()
                else:
                    self.cursor.move(g.x() + clipped[0], g.y() + clipped[1])
            except Exception as exc:
                self.emergency_stop()
                self.problem(f"Pointer backend failed: {exc}")
        if now - self.last_draw > 0.08:
            self.draw_camera(frame, obs)
            self.update_diagnostics(obs, raw, filtered, in_domain)
            self.last_draw = now
        if not self.cal_button.isEnabled() and not self.experiment:
            self._controls()

    def draw_camera(self, frame, obs):
        display = frame.copy()
        points = obs.landmarks
        if points is not None:
            if self.mesh_check.isChecked():
                from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

                for edge in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                    a, b = points[edge.start, :2].astype(int), points[edge.end, :2].astype(int)
                    cv2.line(display, tuple(a), tuple(b), (85, 110, 85), 1, cv2.LINE_AA)
            for center, contour in ((468, [469, 470, 471, 472]), (473, [474, 475, 476, 477])):
                cv2.polylines(display, [points[contour, :2].astype(np.int32)], True, (0, 220, 250), 1)
                xy = points[center, :2].astype(int)
                cv2.drawMarker(display, tuple(xy), (0, 180, 255), cv2.MARKER_CROSS, 9, 1)
            for label, corners in zip(self.eye_labels, ((33, 133), (362, 263))):
                p = points[list(corners), :2]
                center = p.mean(axis=0)
                radius = max(20, int(np.linalg.norm(p[1] - p[0]) * 0.85))
                x, y = center.astype(int)
                crop = display[max(0, y - radius // 2) : y + radius // 2, max(0, x - radius) : x + radius]
                self.set_image(label, crop, mirror=self.mirror_check.isChecked())
        else:
            for label in self.eye_labels:
                label.setText("No iris detected")
        self.set_image(self.video_label, display, mirror=self.mirror_check.isChecked())

    @staticmethod
    def set_image(label, bgr, mirror=False):
        if bgr.size == 0:
            return
        if mirror:
            bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )

    def update_diagnostics(self, obs, raw, filtered, in_domain):
        fps = (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 0.001)
        state = "VALID" if obs.valid else "REJECTED: " + "; ".join(obs.reasons)
        if obs.valid and self.model and not in_domain:
            state = "OUTSIDE CALIBRATED HEAD RANGE — mouse frozen; return to original posture"
        m = obs.metrics
        text = f"Tracking: {state}\n"
        text += (
            f"Processed FPS {fps:.1f} | inference median {np.median(self.inference_times):.1f} ms | "
            f"capture-read→UI median {np.median(self.ages):.1f} ms (excludes sensor/display latency)\n"
        )
        if m:
            text += (
                f"Head yaw {m['yaw_deg']:+.1f}°  pitch {m['pitch_deg']:+.1f}°  roll {m['roll_deg']:+.1f}° | "
                f"approx camera depth {m['distance_mm'] / 10:.1f} cm\n"
                f"Eye widths R/L {m['right_eye_px']:.1f}/{m['left_eye_px']:.1f} px | "
                f"opening {m['right_open']:.3f}/{m['left_open']:.3f} | blink {m['blink']:.2f}\n"
            )
            text += f"Face center (normalized image): ({obs.features[7]:.3f}, {obs.features[8]:.3f}) | apparent width {m['face_scale']:.3f}\n"
        if raw is not None:
            text += f"Raw gaze ({raw[0]:.0f}, {raw[1]:.0f}) px | filtered ({filtered[0]:.0f}, {filtered[1]:.0f}) px | "
        text += "MOUSE ARMED" if self.cursor.armed else "Mouse disarmed"
        self.diagnostics.setText(text)

    def watchdog(self):
        if self.cursor.armed:
            try:
                if time.monotonic() - self.last_seen > 0.3 or self.cursor.emergency_requested():
                    self.emergency_stop()
            except Exception:
                self.emergency_stop()
        if self.target_window and self.experiment and time.monotonic() - self.last_seen > 1:
            self.target_window.message = "Camera stalled — cancel and restart camera"
            self.target_window.update()

    def start_experiment(self, kind):
        if not self.latest or not self.latest.valid:
            self.problem("Get valid face and eye tracking before starting an experiment.")
            return
        self.emergency_stop()
        self.collect_settings()
        self.experiment = Experiment(kind, self.settings)
        self.target_window = TargetWindow(self.screen(), kind)
        self.target_window.cancelled.connect(self.cancel_experiment)
        self.target_window.started.connect(self.refresh_target)
        self.target_window.skipped.connect(self.skip_target)
        self.target_window.painted.connect(self.target_painted)
        self._controls()

    def target_painted(self, timestamp):
        if self.experiment:
            self.experiment.begin_target(timestamp)

    def refresh_target(self):
        if not self.target_window or not self.experiment:
            return
        e = self.experiment
        self.target_window.target = e.target
        self.target_window.count = f"{e.index + 1}/{len(e.schedule)}"
        self.target_window.message = e.message
        self.target_window.progress = e.progress
        self.target_window.update()

    def skip_target(self):
        self.experiment.skip()
        if self.experiment.done:
            self.finish_experiment()
        else:
            self.refresh_target()

    def run_metadata(self):
        return {
            "camera": self.camera_info,
            "screen": self.screen_info(),
            "compatibility": compatibility(self.settings, self.camera_info, self.screen_info()),
            "inference_ms_median": float(np.median(self.inference_times)) if self.inference_times else None,
            "capture_read_to_ui_ms_median": float(np.median(self.ages)) if self.ages else None,
            "seed": self.experiment.seed if self.experiment else None,
        }

    def finish_experiment(self):
        e = self.experiment
        metadata = self.run_metadata()
        report = None
        try:
            if e.kind == "calibration":
                candidate = GazeModel().fit(e.visits)
                self.model = candidate
                self.profile_context = metadata["compatibility"]
                report = candidate.cv
                self.results.appendPlainText(
                    "\nCalibration fitted. Run independent validation next.\n" + json.dumps(report, indent=2)
                )
            else:
                report = validation_report(
                    e.visits,
                    self.model,
                    self.screen_size(),
                    self.settings.screen_width_mm,
                    self.settings.viewing_distance_mm,
                )
                self.results.appendPlainText("\nINDEPENDENT VALIDATION\n" + json.dumps(report, indent=2))
            folder = save_session(
                e.kind, e.visits, self.settings, metadata, report, self.model, trace=e.trace
            )
            if e.kind == "validation" and report.get("targets"):
                from .plots import save_validation_plot

                save_validation_plot(folder / "validation.png", report, self.screen_size())
            self.session_paths.append(folder)
            self.results.appendPlainText(f"\nSaved numeric samples and report: {folder}")
            self.status.setText(
                f"{e.kind.capitalize()} finished. Results saved to {folder.name}. Mouse disarmed."
            )
            if e.kind == "calibration":
                atomic_json(DATA_DIR / "profiles" / "latest.json", self.profile_payload())
        except Exception as exc:
            self.results.appendPlainText(f"\nExperiment error: {exc}")
            try:
                folder = save_session(
                    e.kind,
                    e.visits,
                    self.settings,
                    metadata,
                    {"error": str(exc)},
                    outcome="failed",
                    trace=e.trace,
                )
                self.results.appendPlainText(f"Failed run data preserved at {folder}")
            except Exception as save_error:
                self.results.appendPlainText(f"Could not preserve run: {save_error}")
            self.status.setText(f"Experiment failed: {exc}")
        self.close_target()
        self.experiment = None
        self.filter.reset()
        self._controls()

    def close_target(self):
        if self.target_window:
            target = self.target_window
            self.target_window = None
            target.finishing = True
            target.close()

    def cancel_experiment(self):
        if self.experiment:
            try:
                e = self.experiment
                if not e.done and e.onset is not None:
                    e._finish("cancelled")
                folder = save_session(
                    e.kind, e.visits, self.settings, self.run_metadata(), outcome="cancelled", trace=e.trace
                )
                self.results.appendPlainText(f"\nCancelled run data saved: {folder}")
            except Exception as exc:
                self.results.appendPlainText(f"Could not save cancelled run: {exc}")
            self.experiment = None
        self.close_target()
        self.filter.reset()
        self._controls()

    def toggle_preview(self, checked):
        if self.preview_window:
            self.preview_window.close()
            self.preview_window = None
        if checked and self.model:
            self.preview_window = PreviewWindow(self.screen())

    def toggle_mouse(self):
        if self.cursor.armed:
            self.emergency_stop()
            return
        if self.screen().devicePixelRatio() != 1:
            self.problem(
                "Native pointer currently requires display scale 100%. Gaze preview and experiments support scaling."
            )
            return
        if not self.latest or not self.latest.valid or time.monotonic() - self.last_seen > 0.25:
            self.problem("Valid current eye tracking is required to arm the mouse.")
            return
        if not self.model.in_domain(self.latest.features):
            self.problem("Head outside calibrated range. Return to the calibration posture or recalibrate.")
            return
        try:
            self.filter.reset()
            self.cursor.arm()
            self.status.setText(
                "MOUSE ARMED. Global ESC or physical mouse at desktop upper-left stops movement."
            )
        except Exception as exc:
            self.problem(str(exc))
        self._controls()

    def emergency_stop(self):
        was_armed = self.cursor.armed
        try:
            self.cursor.disarm()
        except Exception:
            self.cursor.armed = False
        self.filter.reset()
        if hasattr(self, "preview_check"):
            self.preview_check.setChecked(False)
        if was_armed:
            self.status.setText("Mouse disarmed.")
        self._controls()

    def update_filter(self):
        self.settings.min_cutoff = self.inputs["min_cutoff"].value()
        self.settings.filter_beta = self.inputs["filter_beta"].value()
        self.filter = OneEuro(self.settings.min_cutoff, self.settings.filter_beta)

    def screen_changed(self):
        self.emergency_stop()
        self.model = self.profile_context = None
        self.status.setText("Monitor changed. Calibrate again or load a matching profile.")
        self._controls()

    def choose_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Camera intrinsics", str(DATA_DIR), "JSON (*.json)")
        if path:
            try:
                data = json.loads(Path(path).read_text())
                if (
                    np.asarray(data["camera_matrix"]).shape != (3, 3)
                    or "distortion" not in data
                    or "image_size" not in data
                ):
                    raise ValueError("Expected camera_matrix, distortion and image_size")
                self.settings.intrinsics_path = str(Path(path).resolve())
                self.intrinsics_label.setText(path)
            except Exception as exc:
                self.problem(f"Invalid intrinsics: {exc}")

    def clear_intrinsics(self):
        self.settings.intrinsics_path = ""
        self.intrinsics_label.setText("No measured intrinsics — approximate FOV used")

    def profile_payload(self):
        return {
            "schema_version": 1,
            "created_utc": utc_now(),
            "compatibility": self.profile_context,
            "settings": self.settings.to_dict(),
            "model": self.model.to_dict(),
        }

    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save calibration profile", str(DATA_DIR / "profiles" / "calibration.json"), "JSON (*.json)"
        )
        if path:
            try:
                atomic_json(path, self.profile_payload())
                self.status.setText(f"Profile saved: {path}")
            except Exception as exc:
                self.problem(str(exc))

    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration profile", str(DATA_DIR / "profiles"), "JSON (*.json)"
        )
        if not path:
            return
        self.emergency_stop()
        try:
            payload = json.loads(Path(path).read_text())
            current = compatibility(self.settings, self.camera_info, self.screen_info())
            if payload.get("compatibility") != current:
                raise ValueError(
                    "Camera, model, optics, resolution or monitor differs from this profile. Restore its settings or recalibrate."
                )
            candidate = GazeModel.from_dict(payload["model"])
            self.model, self.profile_context = candidate, current
            self.status.setText(
                "Profile loaded. Validate again: person, lighting, camera mounting and posture may have changed."
            )
            self.filter.reset()
            self._controls()
        except Exception as exc:
            self.problem(f"Cannot load profile: {exc}")

    def open_data(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATA_DIR)))

    def problem(self, text):
        QMessageBox.warning(self, "Eye Mouse lab", text)

    def closeEvent(self, event):
        self.emergency_stop()
        if self.experiment:
            self.cancel_experiment()
            self.close_target()
        if self.worker:
            self.worker.requestInterruption()
            if not self.worker.wait(3000):
                self.status.setText("Waiting for camera worker to release the device; close again shortly.")
                event.ignore()
                return
        event.accept()
