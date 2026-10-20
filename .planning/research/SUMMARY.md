# Research and implementation constraints

- MediaPipe Tasks Face Landmarker supplies 478 landmarks, blendshapes and facial transformation
  matrices. It supplies face geometry, not a calibrated screen gaze point. Personal mapping is needed.
  https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
- Iris positions are represented in local eye coordinates to reduce head-roll coupling. Include face
  translation, scale and head rotation in mapping; avoid claiming full geometric gaze ray recovery.
- Monocular depth depends on focal length and assumed/known anatomy. Label IPD-based depth as
  approximate. A chessboard calibrates camera intrinsics and distortion, not personal gaze accuracy.
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Transformation matrices provide head pose relative to a canonical face; translation is not a
  personalized metric measurement. Do not label it as exact face distance.
- Use target-grouped CV instead of splitting adjacent video frames. Report independent validation
  error and frame/visit dropout; smoothing has latency and must be evaluated separately.
