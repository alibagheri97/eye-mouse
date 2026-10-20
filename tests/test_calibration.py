import copy

import numpy as np
import pytest

from eye_mouse.calibration import GazeModel, Visit, validation_report


def features(x, y, shift=0.0):
    return np.array(
        [
            (x - 0.5) * 0.5,
            (y - 0.5) * 0.4,
            (x - 0.5) * 0.48,
            (y - 0.5) * 0.42,
            shift,
            shift / 2,
            0,
            0.5 + shift / 5,
            0.45,
            -1.6,
        ]
    )


def calibration_visits():
    rng = np.random.default_rng(79)
    visits = []
    axis = np.linspace(0.06, 0.94, 5)
    for repeat in range(2):
        for group, (x, y) in enumerate((x, y) for y in axis for x in axis):
            samples = [features(x, y, repeat * 0.03) + rng.normal(0, 0.0004, 10) for _ in range(24)]
            visits.append(
                Visit([float(x), float(y)], group, np.array(samples).tolist(), np.arange(24).tolist(), 27, 3)
            )
    return visits


def test_mapping_generalizes_to_unseen_targets_and_roundtrips():
    model = GazeModel().fit(calibration_visits())
    for x, y in ((0.17, 0.39), (0.61, 0.83), (0.35, 0.15)):
        np.testing.assert_allclose(model.predict(features(x, y, 0.01)), [x, y], atol=0.025)
    restored = GazeModel.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.predict(features(0.3, 0.7)), model.predict(features(0.3, 0.7)))
    assert restored.in_domain(features(0.5, 0.5))
    outside = features(0.5, 0.5)
    outside[4] = 0.7
    assert not restored.in_domain(outside)


def test_all_repeats_of_same_target_stay_in_cv_fold(monkeypatch):
    import eye_mouse.calibration as calibration

    original = calibration.solve
    training_sizes = []

    def spy(x, y, alpha):
        training_sizes.append(len(x))
        # Each target has exactly two visits. A grouped fold must keep them together.
        unique, counts = np.unique(y, axis=0, return_counts=True)
        assert len(unique) * 2 == len(y)
        assert np.all(counts == 2)
        return original(x, y, alpha)

    monkeypatch.setattr(calibration, "solve", spy)
    GazeModel().fit(calibration_visits())
    assert training_sizes.count(40) == 20
    assert training_sizes[-1] == 50


def test_reject_insufficient_or_degenerate_calibration():
    with pytest.raises(ValueError, match="distinct"):
        GazeModel().fit(calibration_visits()[:8])
    visits = calibration_visits()
    for visit in visits:
        visit.samples = [features(0.5, 0.5).tolist()] * 24
    with pytest.raises(ValueError, match="iris variation"):
        GazeModel().fit(visits)


def test_profile_rejects_corrupt_numeric_arrays():
    payload = GazeModel().fit(calibration_visits()).to_dict()
    for key, value in (("std", [0] * 10), ("weights", [[1, 2]]), ("mean", [float("nan")] * 10)):
        corrupt = copy.deepcopy(payload)
        corrupt[key] = value
        with pytest.raises(ValueError):
            GazeModel.from_dict(corrupt)


def test_validation_is_raw_unclipped_and_distinguishes_accuracy_from_precision():
    class ConstantModel:
        def predict(self, _):
            return np.array([1.1, 0.5])

    visit = Visit([0.9, 0.5], 0, [[0] * 10] * 20, list(range(20)), 25, 5)
    before = copy.deepcopy(visit)
    report = validation_report([visit], ConstantModel(), [1000, 800], 300, 600)
    assert report["mean_frame_error_px"] == pytest.approx(200)
    assert report["mean_fixation_accuracy_px"] == pytest.approx(200)
    assert report["mean_fixation_precision_rms_px"] == pytest.approx(0)
    assert report["rejected_fraction"] == pytest.approx(0.2)
    assert report["approx_mean_angle_deg"] > 5
    assert visit == before


def test_validation_records_skipped_targets_as_partial():
    model = GazeModel().fit(calibration_visits())
    good = Visit([0.17, 0.39], 0, [features(0.17, 0.39).tolist()] * 20, list(range(20)), 20, 0)
    missing = Visit([0.83, 0.61], 1, [], [], 100, 100, "timeout")
    report = validation_report([good, missing], model, [1920, 1080], 310, 600)
    assert report["status"] == "partial"
    assert report["completed_targets"] == 1
    assert report["scheduled_targets"] == 2
