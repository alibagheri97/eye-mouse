# Experimental lab delivery

Implemented the complete plain desktop lab, described in README.md. Local camera inference feeds
binocular iris/head features, repeated calibration, grouped ridge selection, independent validation,
adaptive preview/native cursor and durable experiment exports. Camera optics can be measured using
the optional chessboard utility. UI refinement is explicitly deferred.

## Validation
- `uv run pytest -q`: 21 passed, including offscreen GUI keyboard/cancellation and full synthetic
  fit → profile → independent validation → JSON/CSV/plot export integration.
- `uv run ruff check eye_mouse tests tools`: passed.
- `uv sync --frozen --python 3.12`: passed.
- Camera doctor: 20/20 frames detected complete face geometry and passed quality checks; mean
  inference 12.31 ms in this run. This says nothing about point-of-regard accuracy.
- Desktop camera test: 1280×720 at actual 10 FPS, live mesh/eye views and diagnostics, fullscreen
  acquisition, Escape cancellation, device release and a screenshot after clearing camera pixels.
- Native cursor: arming, actual pointer movement, global Escape delivery, disarming and pointer
  restoration verified by `tools/cursor_check.py`.

## Fixes found during verification
- Explicit top-level window deletion on close avoids a PySide6 shutdown crash in this environment.
- Contiguous fixation acquisition extends until minimum frame count is met, supporting a real 10 FPS
  camera without the fixed sliding-window timeout failure.
- Plain Escape grabs include CapsLock/NumLock states while avoiding desktop shortcut modifier grabs.
- Rejected/blink/timeout data are excluded from fitting; independent errors are unclipped/unsmoothed.

## Boundaries
No actual personal gaze calibration/accuracy result is claimed. No measured intrinsics were produced
without a printed board. Native movement supports X11 at 100% scaling; other displays retain the lab
and preview. No camera images/video are persisted. Numeric runtime-check sessions are local and ignored.
