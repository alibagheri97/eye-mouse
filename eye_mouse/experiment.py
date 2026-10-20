"""Deterministic fixation acquisition state machine, independent of the GUI."""

from collections import deque

import numpy as np

from .calibration import Visit


class Experiment:
    def __init__(self, kind, settings, seed=20261021):
        self.kind, self.settings, self.seed = kind, settings, seed
        rng = np.random.default_rng(seed)
        self.schedule = []
        if kind == "calibration":
            axis = np.linspace(0.06, 0.94, settings.calibration_grid)
            targets = [(float(x), float(y)) for y in axis for x in axis]
            for _ in range(settings.calibration_passes):
                for index in rng.permutation(len(targets)):
                    self.schedule.append((targets[index], int(index)))
        else:
            # Off-grid coordinates never appear in the default calibration grid.
            targets = [(x, y) for y in (0.17, 0.39, 0.61, 0.83) for x in (0.17, 0.39, 0.61, 0.83)]
            self.schedule = [(targets[i], int(i)) for i in rng.permutation(len(targets))]
        self.index = 0
        self.visits = []
        self.trace = []
        self.buffer = deque()
        self.onset = None
        self.attempted = self.rejected = 0
        self.message = "Look at the dot. Keep your head comfortable."

    @property
    def done(self):
        return self.index >= len(self.schedule)

    @property
    def target(self):
        return self.schedule[self.index][0] if not self.done else (0.5, 0.5)

    @property
    def progress(self):
        if len(self.buffer) < 2:
            return 0.0
        return min(1.0, (self.buffer[-1][0] - self.buffer[0][0]) / self.settings.sample_seconds)

    def begin_target(self, timestamp):
        """Called after the GUI has painted the target, never at construction time."""
        if self.onset is None:
            self.onset = timestamp

    def update(self, observation):
        if self.done or self.onset is None:
            return False
        now = observation.timestamp
        elapsed = now - self.onset
        if elapsed < self.settings.settle_seconds:
            self.message = "Settle on the dot…"
            return False
        if elapsed >= self.settings.target_timeout:
            self._finish("timeout")
            return True
        self.attempted += 1
        self.trace.append(
            {
                "visit": self.index,
                "target": list(self.target),
                "timestamp": now,
                "valid": observation.valid,
                "reasons": list(observation.reasons),
                "features": observation.features.tolist() if observation.features is not None else None,
                "metrics": observation.metrics,
                "inference_ms": observation.inference_ms,
            }
        )
        if not observation.valid:
            self.rejected += 1
            self.buffer.clear()
            self.message = "; ".join(observation.reasons)
        else:
            # Discontinuous/stalled camera frames never count as sustained fixation.
            if self.buffer and now - self.buffer[-1][0] > 0.2:
                self.buffer.clear()
            self.buffer.append((now, observation.features.copy()))
            # Keep the contiguous fixation until BOTH duration and frame count are sufficient.
            # A fixed sliding window would never reach min_samples on a dim/slow webcam.
            samples = np.array([x[1] for x in self.buffer])
            if len(samples) >= 6:
                med = np.median(samples[:, :4], axis=0)
                jitter = float(np.max(np.median(abs(samples[:, :4] - med), axis=0)))
                head_range = np.ptp(samples[:, 4:6], axis=0)
                if jitter > self.settings.fixation_jitter or np.any(head_range > np.deg2rad(8)):
                    self.rejected += len(self.buffer)
                    self.buffer.clear()
                    self.message = "Unstable fixation — keep looking at the dot"
                    return False
            self.message = f"Collecting stable gaze: {self.progress:.0%}"
            if self.progress >= 1 and len(self.buffer) >= self.settings.min_samples:
                self._finish("accepted")
                return True
        return False

    def _finish(self, status):
        target, group = self.schedule[self.index]
        self.visits.append(
            Visit(
                list(target),
                group,
                [x[1].tolist() for x in self.buffer] if status == "accepted" else [],
                [x[0] for x in self.buffer] if status == "accepted" else [],
                self.attempted,
                self.attempted - len(self.buffer) if status == "accepted" else self.attempted,
                status,
            )
        )
        self.index += 1
        self.onset = None
        self.buffer.clear()
        self.attempted = self.rejected = 0

    def skip(self):
        if not self.done:
            self._finish("skipped")
