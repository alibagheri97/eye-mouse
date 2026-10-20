"""Target-grouped model selection and independent fixation-level validation."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import FEATURE_NAMES


def design(z):
    z = np.atleast_2d(z)
    # Compact nonlinear basis: eye squares and eye × head-angle interactions.
    return np.column_stack([z, z[:, :4] ** 2, *[z[:, i] * z[:, j] for i in range(4) for j in (4, 5)]])


def solve(x, y, alpha):
    xmean, ymean = x.mean(axis=0), y.mean(axis=0)
    xc = x - xmean
    weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - ymean))
    return weights, ymean - xmean @ weights


@dataclass
class Visit:
    target: list[float]
    group: int
    samples: list[list[float]]
    timestamps: list[float]
    attempted: int
    rejected: int
    status: str = "accepted"

    def to_dict(self):
        return asdict(self)


class GazeModel:
    VERSION = 1

    def fit(self, visits):
        accepted = [v for v in visits if v.status == "accepted" and v.samples]
        x = np.array([np.median(v.samples, axis=0) for v in accepted])
        y = np.array([v.target for v in accepted])
        groups = np.array([v.group for v in accepted])
        unique = np.unique(groups)
        if len(unique) < 9 or len(x) < 18:
            raise ValueError("Need at least 9 distinct calibration targets and 18 accepted visits")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite calibration samples")
        if np.min(np.ptp(y, axis=0)) < 0.6:
            raise ValueError("Calibration must cover both screen axes")
        if np.linalg.norm(np.std(x[:, :4], axis=0)) < 0.005:
            raise ValueError("Insufficient iris variation; check eye tracking and look at each target")
        folds = np.random.default_rng(42).permutation(unique)
        scores = {}
        for alpha in (0.1, 1.0, 10.0, 100.0):
            errors = []
            for held in np.array_split(folds, min(5, len(unique))):
                train = ~np.isin(groups, held)
                mean, std = x[train].mean(axis=0), np.maximum(x[train].std(axis=0), 0.01)
                a, b = solve(design((x[train] - mean) / std), y[train], alpha)
                prediction = design((x[~train] - mean) / std) @ a + b
                errors.extend(np.linalg.norm(prediction - y[~train], axis=1).tolist())
            scores[str(alpha)] = float(np.mean(errors))
        self.alpha = float(min(scores, key=scores.get))
        self.mean, self.std = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
        self.weights, self.intercept = solve(design((x - self.mean) / self.std), y, self.alpha)
        # Head pose / position envelope only. Gaze can legitimately extend between targets.
        self.low, self.high = x[:, 4:].min(axis=0), x[:, 4:].max(axis=0)
        self.margin = np.maximum((self.high - self.low) * 0.35, [0.10, 0.10, 0.10, 0.035, 0.035, 0.12])
        self.cv = {
            "selection_error_normalized": scores,
            "selected_alpha": self.alpha,
            "accepted_visits": len(accepted),
            "distinct_targets": len(unique),
            "note": "Grouped by target; model-selection diagnostic, not independent accuracy.",
        }
        return self

    def predict(self, features):
        return (design((np.asarray(features) - self.mean) / self.std) @ self.weights + self.intercept)[0]

    def in_domain(self, features):
        head = np.asarray(features)[4:]
        return bool(np.all(head >= self.low - self.margin) and np.all(head <= self.high + self.margin))

    def to_dict(self):
        return {
            "version": self.VERSION,
            "feature_names": FEATURE_NAMES,
            "alpha": self.alpha,
            **{
                k: getattr(self, k).tolist()
                for k in ("mean", "std", "weights", "intercept", "low", "high", "margin")
            },
            "cv": self.cv,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != cls.VERSION or data.get("feature_names") != FEATURE_NAMES:
            raise ValueError("Incompatible profile feature schema")
        obj = cls()
        expected = {
            "mean": (10,),
            "std": (10,),
            "weights": (22, 2),
            "intercept": (2,),
            "low": (6,),
            "high": (6,),
            "margin": (6,),
        }
        for name, shape in expected.items():
            value = np.asarray(data[name], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid profile array: {name}")
            setattr(obj, name, value)
        if np.any(obj.std <= 0) or np.any(obj.margin < 0):
            raise ValueError("Invalid profile scale")
        obj.alpha, obj.cv = float(data["alpha"]), data["cv"]
        return obj


def validation_report(visits, model, screen_size, screen_width_mm, distance_mm):
    size = np.asarray(screen_size)
    rows, all_errors = [], []
    accepted = [v for v in visits if v.status == "accepted" and v.samples]
    for v in accepted:
        predictions = np.array([model.predict(x) for x in v.samples]) * size
        target = np.array(v.target) * size
        errors = np.linalg.norm(predictions - target, axis=1)
        center = np.mean(predictions, axis=0)
        precision = float(np.sqrt(np.mean(np.sum((predictions - center) ** 2, axis=1))))
        steps = np.diff(predictions, axis=0)
        rows.append(
            {
                "target": v.target,
                "mean_prediction_px": center.tolist(),
                "accuracy_px": float(np.linalg.norm(center - target)),
                "frame_error_mean_px": float(errors.mean()),
                "frame_error_p95_px": float(np.percentile(errors, 95)),
                "precision_rms_px": precision,
                "successive_difference_rms_px": float(np.sqrt(np.mean(np.sum(steps**2, axis=1))))
                if len(steps)
                else 0,
                "samples": len(v.samples),
            }
        )
        all_errors.extend(errors.tolist())
    attempts = sum(v.attempted for v in visits)
    rejects = sum(v.rejected for v in visits)
    if not rows:
        return {
            "status": "no accepted validation visits",
            "attempted_frames": attempts,
            "rejected_frames": rejects,
            "completed_targets": 0,
        }
    errors = np.array(all_errors)

    # Square-pixel assumption; physical display width must be entered by the operator.
    def angle(px):
        return math.degrees(2 * math.atan(px * screen_width_mm / size[0] / (2 * distance_mm)))

    return {
        "status": "complete" if len(rows) == len(visits) else "partial",
        "completed_targets": len(rows),
        "scheduled_targets": len(visits),
        "mean_fixation_accuracy_px": float(np.mean([r["accuracy_px"] for r in rows])),
        "mean_frame_error_px": float(errors.mean()),
        "median_frame_error_px": float(np.median(errors)),
        "p95_frame_error_px": float(np.percentile(errors, 95)),
        "mean_fixation_precision_rms_px": float(np.mean([r["precision_rms_px"] for r in rows])),
        "approx_mean_angle_deg": angle(float(errors.mean())),
        "approx_p95_angle_deg": angle(float(np.percentile(errors, 95))),
        "attempted_frames": attempts,
        "rejected_frames": rejects,
        "rejected_fraction": rejects / max(attempts, 1),
        "note": "Raw unclipped gaze; no smoothing. Angular conversion assumes entered screen width and distance.",
        "targets": rows,
    }
