import math

import numpy as np
import pytest

from viam_arm_policy.policy import SAFETY_PROFILES
from viam_arm_policy.trajectory_conditioner import (
    condition_trajectory,
    condition_trajectory_toppra,
)

PERIOD_S = 0.05


def _actions(value=0.0):
    actions = np.zeros((32, 7), dtype=np.float64)
    actions[:, :6] = value
    actions[:, 6] = 0.25
    return actions


def _history(value=0.0):
    sample = np.full(6, value, dtype=np.float64)
    return [sample.copy(), sample.copy()]


def test_valid_trajectory_is_an_exact_noop():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions()

    conditioned, diagnostics = condition_trajectory(
        raw, _history(), profile, period_s=PERIOD_S
    )

    np.testing.assert_array_equal(conditioned, raw)
    assert diagnostics["conditioned"] is False
    assert diagnostics["max_path_deviation_rad"] == 0.0


def test_time_parameterization_anchors_to_measured_state_and_preserves_waypoints():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions(0.4)

    conditioned, diagnostics = condition_trajectory(
        raw, _history(), profile, period_s=PERIOD_S
    )

    first_acceleration = conditioned[0, :6] / (PERIOD_S * PERIOD_S)
    assert np.all(np.abs(first_acceleration) <= profile.acceleration_limits)
    assert any(np.array_equal(row, raw[0]) for row in conditioned)
    assert conditioned.shape[0] > raw.shape[0]
    assert diagnostics["conditioned"] is True
    assert diagnostics["max_path_deviation_rad"] <= math.radians(5.0)


def test_toppra_preserves_path_without_stopping_at_every_waypoint():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions(0.4)
    stopped, _ = condition_trajectory(raw, _history(), profile, period_s=PERIOD_S)

    conditioned, diagnostics = condition_trajectory_toppra(
        raw, _history(), profile, period_s=PERIOD_S
    )

    assert conditioned.shape[0] < stopped.shape[0]
    np.testing.assert_allclose(conditioned[-1, :6], raw[-1, :6], atol=1e-9)
    metrics = diagnostics["conditioned_metrics"]
    assert max(metrics["max_abs_velocity_rad_s"]) <= max(profile.velocity_limits)
    assert max(metrics["max_abs_acceleration_rad_s2"]) <= max(profile.acceleration_limits)
    assert diagnostics["max_path_deviation_rad"] <= math.radians(5.0)


def test_dynamic_limit_boundaries_are_accepted_without_conditioning():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions()
    raw[0, 0] = profile.acceleration_limits[0] * PERIOD_S**2
    raw[1:, 0] = raw[0, 0]

    conditioned, diagnostics = condition_trajectory(
        raw, _history(), profile, period_s=PERIOD_S
    )

    np.testing.assert_array_equal(conditioned, raw)
    assert diagnostics["conditioned"] is False


def test_excessive_measured_velocity_fails_closed():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    history = _history()
    history[1][0] = (profile.velocity_limits[0] + 0.1) * PERIOD_S

    with pytest.raises(ValueError, match="measured joint history"):
        condition_trajectory(_actions(), history, profile, period_s=PERIOD_S)


def test_conditioning_horizon_limit_fails_closed_instead_of_dropping_waypoints():
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]

    with pytest.raises(ValueError, match="maximum conditioned horizon"):
        condition_trajectory(
            _actions(0.4),
            _history(),
            profile,
            period_s=PERIOD_S,
            max_conditioned_steps=32,
        )


def test_toppra_gripper_changes_where_the_raw_chunk_changes_it_not_earlier():
    # Descend for 10 steps, dwell, and close the gripper two steps into the
    # dwell, as recorded grasps do. Removing the dwell from the joint path must
    # not move the close up into the descent.
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions()
    raw[:10, 1] = np.linspace(0.05, 0.5, 10)
    raw[10:, 1] = 0.5
    raw[:, 6] = 0.0
    raw[12:, 6] = 1.0

    conditioned, diagnostics = condition_trajectory_toppra(
        raw, _history(), profile, period_s=PERIOD_S
    )

    assert diagnostics["conditioned"] is True
    closed = conditioned[:, 6] >= 0.5
    assert closed.any()
    np.testing.assert_allclose(conditioned[closed, 1], 0.5, atol=1e-6)
    assert conditioned[-1, 6] == 1.0


def test_toppra_gripper_follows_progress_along_a_moving_path():
    # A gripper change partway through continuous motion happens once the arm
    # has travelled that far, not at an evenly spaced time.
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions()
    raw[:, 1] = np.linspace(0.04, 1.28, 32)
    raw[:, 6] = 0.0
    raw[16:, 6] = 1.0

    conditioned, _ = condition_trajectory_toppra(
        raw, _history(), profile, period_s=PERIOD_S
    )

    closed = conditioned[:, 6] >= 0.5
    assert closed.any() and not closed.all()
    assert conditioned[closed, 1].min() >= raw[16, 1] - 1e-6
    assert conditioned[~closed, 1].max() < raw[16, 1]


def test_toppra_gripper_change_in_a_jittery_dwell_still_happens():
    # Model dwells are not exact repeats; sub-epsilon jitter is dropped from the
    # path and must not leave the final close out of reach.
    profile = SAFETY_PROFILES[("lite6", "lite6_gripper")]
    raw = _actions()
    raw[:10, 1] = np.linspace(0.05, 0.5, 10)
    raw[10:, 1] = 0.5 + np.tile([0.0, 1e-12], 11)
    raw[:, 6] = 0.0
    raw[30:, 6] = 1.0

    conditioned, _ = condition_trajectory_toppra(
        raw, _history(), profile, period_s=PERIOD_S
    )

    assert conditioned[-1, 6] == 1.0
    assert (conditioned[:-1, 6] == 0.0).all() or conditioned[conditioned[:, 6] >= 0.5, 1].min() >= 0.5 - 1e-6
