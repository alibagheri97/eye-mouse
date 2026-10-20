# Phase 01 plan

1. Implement local camera pipeline, iris geometry, face pose and optics diagnostics.
2. Implement quality-gated randomized calibration, grouped ridge selection, independent validation,
   adaptive filtering and data/profile persistence with compatibility checks.
3. Integrate plain PySide6 controls, fullscreen targets, native cursor arming and emergency stop.
4. Add camera-intrinsics utility, run meaningful numerical tests and exercise inference/GUI/camera.
5. Document exact launch steps and any remaining human validation boundary.

Acceptance: no polished UI work; no fabricated gaze measurements; explicit optics assumptions;
tracking loss and invalid data never move the pointer; held-out visits never fit the model.
