# State

Phase: 01 — Experimental lab
Status: lab implemented and verified; ready for the user's personal calibration and validation
Last activity: 2026-10-21 — 21 tests and runtime checks passed; UI refinement remains deferred.

## Verification boundary
Automated math, camera and GUI checks can verify operation. Human gaze accuracy requires a real
calibration and independent validation; do not claim it from synthetic tests.

## Completed
- 478-landmark face/iris inference, pose, image-space position/scale and estimated depth.
- 50-fixation acquisition, grouped mapping selection, independent 16-target validation and plots.
- Profiles, raw numeric traces, CSV and package/model/settings metadata.
- X11 native movement and plain-Escape global stop, verified with pointer restored afterward.
- Plain desktop controls, camera intrinsics utility, locked environment and launcher.

## Runtime evidence
`artifacts/camera-check.json`, `artifacts/gui-runtime.json`, `artifacts/cursor-runtime.json` (ignored).
The actual camera returned 1280×720 at 10 FPS; sampling supports that observed rate.
Native GUI and offscreen keyboard/profile/export tests passed. No human gaze accuracy was inferred.
