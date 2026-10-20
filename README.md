# Eye Mouse — experimental lab

A local desktop laboratory for webcam gaze tracking and native mouse movement. Plain controls, visible
measurements and repeatable experiments come first. Python, PySide6, OpenCV and MediaPipe Tasks; no
account, API key or cloud inference.

## Run

From this directory:

```bash
./run-lab.sh
```

The launcher uses [uv](https://docs.astral.sh/uv/), Python 3.12, the locked dependencies, and the official
MediaPipe face landmark model (downloaded once, checksum verified). Dependencies/model need internet
on first setup; tracking then works offline. Equivalent manual setup:

```bash
uv sync --frozen --python 3.12
uv run eye-mouse download-model
uv run eye-mouse
```

Native pointer movement currently supports **Linux X11 at 100% display scaling**. The lab, calibration,
validation and gaze preview also work without native pointer support; Wayland intentionally reports
that mouse control is unavailable. Multi-monitor calibration targets the selected monitor only.

## First experiment

1. Mount the webcam firmly near the monitor. Use even face lighting. Keep autofocus/exposure stable
   where your camera's own controls permit it. Choose the camera index and monitor, then **Start camera**.
2. In **Live measurements**, check the 478-point face mesh, both iris centers and eye close-ups. Wait
   for `VALID`. Read the actual capture size and processed FPS; requested camera settings may differ.
3. In **Experiment settings**, enter measured active monitor width and eye-to-screen distance for
   approximate angular errors. Enter your IPD and camera horizontal FOV for approximate camera depth.
   Default values are assumptions, not measurements of your body or hardware.
4. Select **Calibrate**, then press **Space**. Look at each dot's center. The routine randomizes 25
   locations over two passes (50 fixations). It discards the initial settling interval, rejects blinks,
   requires a stable fixation and a minimum number of frames, and waits longer on slow cameras.
   Blink between dots. Stay steady within a fixation; a small comfortable posture change between
   passes helps collect some head variation. **S** skips a target; **Esc** cancels and saves partial data.
5. Select **Validate** for 16 separate, off-grid target locations. Read the results before arming the
   mouse. Recalibrate if systematic error is large or important screen areas could not be acquired.
6. Enable **Gaze preview**, then **Arm mouse** when tracking is valid. Adjust filter cutoff and beta
   to trade stationary jitter against responsiveness. **Global Esc** stops the mouse while armed.
   Moving your physical mouse to the desktop's upper-left also stops it. Mouse buttons remain manual.

Tracking loss, blinks, stale camera data and head positions outside the calibrated envelope freeze
movement. A stalled camera disarms after 300 ms without a fresh processed observation. Calibration and
validation disarm the mouse. Models are invalidated when restarting the camera or changing monitors;
load a matching saved profile or recalibrate. Validate again after loading any profile, because matching
device settings cannot establish that the same person, lighting, mounting or posture has returned.

## Measurements and method

- **Face mesh:** MediaPipe Tasks supplies 478 landmarks including iris landmarks. It supplies face
  geometry; the application learns the mapping to your screen during personal calibration.
- **Eye features:** iris offsets relative to eye corners, expressed in a head-roll-aligned eye frame
  and divided by eye width. Both axes use eye width, avoiding division by changing eyelid opening.
- **Head features:** yaw, pitch and roll from the face transformation matrix, normalized face center
  and log apparent face width. Head rotation is relative to MediaPipe's canonical face; this is not
  a personalized 3D eyeball reconstruction or a physically calibrated visual-axis ray.
- **Mapping:** regularized ridge regression with eye-square and eye–head interaction terms. Each
  stable target visit contributes one median feature vector, preventing fast cameras from dominating.
  Five-fold model selection keeps every repeat of a target in the same fold and fits normalization
  inside the training fold. Calibration CV numbers select regularization; they are not independent
  evidence of gaze accuracy. No validation samples are used for fitting.
- **Distance:** approximate axial camera-to-face depth `focal_px × entered_IPD_mm × cos(yaw) / iris_separation_px`.
  Focal length comes from entered horizontal FOV, or optional measured intrinsics. Anatomical assumptions,
  iris localization, perspective and head rotation introduce error. Mesh z is relative and is never
  presented as metric distance. Eye-to-screen distance is a separate manually entered measurement.
- **Validation:** raw, unclipped, unsmoothed predictions on independent target visits. Reports mean,
  median and 95th-percentile frame error; mean fixation accuracy (distance of mean gaze to target);
  RMS fixation precision (scatter about that mean); successive-sample RMS; per-target details;
  discarded frame fraction and completed/failed target coverage. Angular error is an approximate
  screen-space conversion using entered width/distance and square pixels. It is not a measured
  visual-axis angular error. Partial runs are labeled partial.
- **Timing:** processed FPS, inference time and capture-read-to-UI time. The latter excludes camera
  exposure, buffering before the read, display scanout and user response; it is not end-to-end latency.
- **Filtering:** One Euro adaptive low-pass filter in screen pixels. Validation excludes the filter
  so its smoothing and lag do not obscure model error. Live output is clipped to monitor bounds for
  pointer movement only; exported validation errors retain out-of-bounds predictions.

A consumer RGB webcam cannot guarantee pixel-perfect point-of-regard. Test the actual camera, person,
posture and lighting. Reliable tiny-target selection may require dedicated eye-tracking hardware or
later interaction assistance. This lab makes the current error measurable before UI refinement.

## Measured camera intrinsics (optional)

Stop the lab camera. Print a flat **10 × 7 square chessboard**, giving **9 × 6 inner corners**. Measure
the actual printed square size (the default is 25 mm). Capture at least 12 diverse views across the
image with different tilts and distances, then solve:

```bash
uv run eye-mouse camera-calibrate --camera 0 --width 1280 --height 720 \
  --columns 9 --rows 6 --square-mm 25 --output data/camera-intrinsics.json
```

Space captures, Enter solves, Esc cancels. The tool rejects nearly identical consecutive views and
reports reprojection RMS and per-view errors. Poor coverage can still produce poor intrinsics; a low
reprojection error does not establish gaze accuracy. In the lab, **Load camera intrinsics**, then restart
the camera and recalibrate gaze. The lab undistorts frames before extracting landmarks and requires
the exact calibrated capture resolution. Changing focus/crop invalidates camera calibration.

## Saved evidence and privacy

`data/` is local and ignored by Git. Override its destination with `EYE_MOUSE_DATA=/absolute/path`.

- `settings.json`: latest camera/experiment settings.
- `profiles/latest.json`: most recent fitted gaze model; save named copies from **Results / profiles**.
- `sessions/<UTC timestamp>-<kind>/session.json`: settings, package versions, camera/model identity,
  screen geometry, randomized schedule seed, outcome, per-visit features, acquisition trace including
  rejected observations/reasons, model and report.
- `samples.csv`: accepted per-frame features and timestamps for each retained fixation. Units and
  columns are named in the JSON. Timestamps use monotonic seconds within the acquisition process.
- `validation.png`: target-to-mean-gaze arrows and per-target accuracy/precision plot.

Images/video are processed in memory and are **not recorded**. Numeric face/eye samples and profiles
are personal measurements; remove `data/` to delete the locally saved experiments. No background
service or automatic mouse arming is installed.

## Verification and development

```bash
uv run pytest -q
uv run ruff check eye_mouse tests tools
uv run eye-mouse doctor
uv run eye-mouse doctor --camera 0 --frames 30 --output artifacts/camera-check.json
uv run python tools/runtime_check.py --camera
```

To explicitly check native movement and global Escape, `uv run python tools/cursor_check.py` moves
the pointer a few pixels, verifies the grab, then restores its original position without clicking.

Tests cover geometric invariance, grouped model selection, profile validation, acquisition timing,
blinks/stalls, slow-camera sampling, independent metrics, persistence and GUI keyboard flows. Synthetic
samples exercise algorithms and application integration; they cannot establish human gaze accuracy.
`doctor --camera` and the optional desktop runtime check open the actual camera. The desktop check
saves a GUI screenshot only after camera previews are cleared. Diagnostics live under ignored `artifacts/`.

Implementation: `geometry.py` / `tracker.py` → `experiment.py` → `calibration.py` → `filtering.py` /
`cursor.py`, coordinated by `gui.py`. `storage.py` persists evidence; `camera_calibration.py` measures
optics. Research/implementation state is tracked in `.planning/`.

Primary references: [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python),
[OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).
