# Eye Mouse Experimental Lab

## What This Is
A local desktop research lab using a webcam to estimate screen gaze from iris position, face mesh,
head orientation, face position and apparent scale. Native pointer control follows personal calibration.
The first delivery prioritizes reproducible experiments, understandable controls and measured accuracy.

## Core Value
Measure how well gaze estimation works on the user's actual hardware before optimizing appearance.

## Requirements
### Active
- Face mesh, binocular iris features and head orientation with live diagnostics.
- Explicitly approximate monocular distance, configurable optics and camera intrinsics calibration.
- Repeated, quality-gated screen calibration with independent validation and durable profiles.
- Screen-space error, fixation precision, dropout and latency measurements; exportable evidence.
- Native cursor movement with arming, tracking-loss freeze, emergency stop and adjustable smoothing.

### Out of Scope
- UI polish, remote services, cloud inference, click gestures and claims of pixel-perfect accuracy.

## Context
Greenfield Linux X11 checkout; webcam nodes exist. Python 3.12 isolated with uv.
User specifically requested the experimental lab first and UI refinement later.

## Key Decisions
| Decision | Rationale | Outcome |
|---|---|---|
| PySide6 desktop + MediaPipe Tasks | Native screen targets and real pointer control; all inference local | Implement |
| Ridge mapping from geometric eye/head features | Interpretable, regularized, fast and individually calibrated | Measure |
| Separate calibration and validation visits | Training residuals cannot establish gaze accuracy | Required |
| Inline GSD execution | Keep work within the current agent and explicit user scope | Active |

## Evolution
Update requirements, measured limitations and decisions after real user calibration.

Last updated: 2026-10-21
