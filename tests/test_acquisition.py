import numpy as np

from eye_mouse.experiment import Experiment
from eye_mouse.geometry import Observation
from eye_mouse.settings import Settings


def observation(timestamp, valid=True, jitter=0):
    f = np.zeros(10)
    f[:4] = jitter
    return Observation(timestamp, f, valid=valid, reasons=[] if valid else ["blink"])


def test_target_must_be_painted_and_settle_before_collecting():
    e = Experiment("calibration", Settings())
    e.update(observation(5))
    assert not e.buffer
    e.begin_target(10)
    e.update(observation(10.5))
    assert not e.buffer
    e.update(observation(10.9))
    assert len(e.buffer) == 1


def test_blink_resets_contiguous_fixation_and_does_not_train_on_blink():
    settings = Settings(settle_seconds=0.1, sample_seconds=0.6, min_samples=12)
    e = Experiment("calibration", settings)
    e.begin_target(10)
    for t in np.arange(10.11, 10.55, 0.04):
        e.update(observation(t))
    assert e.index == 0
    e.update(observation(10.56, valid=False))
    assert not e.buffer
    for t in np.arange(10.6, 11.3, 0.04):
        if e.update(observation(t)):
            break
    assert e.index == 1
    assert min(e.visits[0].timestamps) > 10.56
    assert e.visits[0].rejected > 0
    assert e.visits[0].attempted == len(e.visits[0].samples) + e.visits[0].rejected
    assert any(row["reasons"] == ["blink"] for row in e.trace)


def test_unstable_fixations_timeout_instead_of_hanging():
    e = Experiment("validation", Settings(settle_seconds=0.1, target_timeout=1, min_samples=8))
    e.begin_target(10)
    for i, t in enumerate(np.arange(10.11, 11.1, 0.03)):
        if e.update(observation(t, jitter=0.2 * (i % 2))):
            break
    assert e.index == 1
    assert e.visits[0].status == "timeout"
    assert not e.visits[0].samples


def test_validation_targets_are_unseen_and_schedules_reproducible():
    a = Experiment("calibration", Settings())
    b = Experiment("validation", Settings())
    assert a.schedule == Experiment("calibration", Settings()).schedule
    assert len(a.schedule) == 50
    assert len(b.schedule) == 16
    assert not {target for target, _ in a.schedule} & {target for target, _ in b.schedule}


def test_camera_stall_breaks_fixation():
    e = Experiment("calibration", Settings(settle_seconds=0.1))
    e.begin_target(0)
    e.update(observation(0.2))
    e.update(observation(0.24))
    e.update(observation(0.7))
    assert len(e.buffer) == 1


def test_low_frame_rate_extends_fixation_to_reach_minimum_frame_count():
    e = Experiment("calibration", Settings(settle_seconds=0.1, sample_seconds=1.2, min_samples=18))
    e.begin_target(10)
    for t in np.arange(10.11, 13, 0.15):
        if e.update(observation(t)):
            break
    assert e.index == 1
    assert len(e.visits[0].samples) >= 18
    assert e.visits[0].timestamps[-1] - e.visits[0].timestamps[0] > 2


def test_skip_and_cancel_never_become_training_visits():
    e = Experiment("calibration", Settings())
    e.skip()
    assert e.index == 1
    assert e.visits[0].status == "skipped"
    assert not e.visits[0].samples
