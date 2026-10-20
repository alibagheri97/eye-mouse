# Verification

Status: implementation verified; human accuracy experiment pending.

| Requirement | Evidence |
|---|---|
| LAB-01 | Real camera doctor + desktop camera runtime; actual face/iris inference, quality/pose/timing |
| LAB-02 | FOV/IPD controls and explicitly approximate depth; intrinsics utility help/runtime API |
| CAL-01 | Acquisition timing/blink/stall/slow-camera tests; actual fullscreen start/ESC cancellation |
| CAL-02 | Target-grouped CV test, unseen-target recovery, profile corruption/compatibility tests |
| VAL-01 | Raw out-of-bounds error/precision tests; synthetic GUI 16-target validation/report/plot |
| PTR-01 | Real X11 motion and global Escape check; loss/domain/freshness gates in on_frame/watchdog |
| DATA-01 | JSON/CSV atomic storage tests; cancelled actual acquisition trace; no camera image recording |
| RUN-01 | Locked uv install, native/offscreen GUI runs, command help, README workflow |

The camera intrinsics utility needs a real printed chessboard for an actual optics result. Accuracy
numbers from generated test features establish numerical behavior only, not performance on a person.
