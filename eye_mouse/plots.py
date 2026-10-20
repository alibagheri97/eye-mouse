"""Standalone experiment figure, not UI decoration."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_validation_plot(path, report, size):
    rows = report["targets"]
    figure = Figure(figsize=(11, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes, distribution = figure.subplots(1, 2)
    target = np.array([r["target"] for r in rows]) * size
    predicted = np.array([r["mean_prediction_px"] for r in rows])
    axes.scatter(*target.T, marker="+", label="Target", color="black")
    axes.scatter(*predicted.T, marker="o", label="Mean raw gaze", color="tab:blue")
    for a, b in zip(target, predicted):
        axes.annotate("", b, a, arrowprops={"arrowstyle": "->", "color": "tab:blue"})
    axes.set(
        xlim=(0, size[0]),
        ylim=(size[1], 0),
        xlabel="Screen x (px)",
        ylabel="Screen y (px)",
        title="Independent validation: target → mean gaze",
    )
    axes.set_aspect("equal")
    axes.legend(fontsize=8)
    errors = [r["accuracy_px"] for r in rows]
    precision = [r["precision_rms_px"] for r in rows]
    distribution.bar(np.arange(len(rows)) - 0.18, errors, width=0.36, label="Fixation accuracy error")
    distribution.bar(np.arange(len(rows)) + 0.18, precision, width=0.36, label="RMS fixation precision")
    distribution.set(
        xlabel="Validation target (acquisition order)",
        ylabel="Pixels",
        title=f"Mean frame error {report['mean_frame_error_px']:.1f}px; p95 {report['p95_frame_error_px']:.1f}px",
    )
    distribution.legend(fontsize=8)
    figure.suptitle("Raw, unclipped, unsmoothed predictions • smaller error is better")
    figure.savefig(path, dpi=160)
