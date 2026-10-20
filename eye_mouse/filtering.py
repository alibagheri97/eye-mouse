import math

import numpy as np


class OneEuro:
    """Adaptive low-pass filter. Inputs and tuning use screen pixels and seconds."""

    def __init__(self, min_cutoff=1.2, beta=0.015):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.reset()

    def reset(self):
        self.value = self.previous = self.derivative = None
        self.time = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def update(self, value, timestamp):
        value = np.asarray(value, dtype=float)
        if self.time is None or timestamp - self.time > 0.25:
            self.value = self.previous = value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = max(timestamp - self.time, 0.001)
            speed = (value - self.previous) / dt
            a = self.alpha(1.0, dt)
            self.derivative += a * (speed - self.derivative)
            cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative)
            self.value += self.alpha(cutoff, dt) * (value - self.value)
            self.previous = value.copy()
        self.time = timestamp
        return self.value.copy()
