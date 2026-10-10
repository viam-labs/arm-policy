import math

import numpy as np
import pytest

from models.policy import SAFETY_PROFILES
from models.trajectory_conditioner import (
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
    assert max(diagnostics["conditioned_metrics"]["max_abs_velocity_rad_s"]) <= 0.5
    assert max(diagnostics["conditioned_metrics"]["max_abs_acceleration_rad_s2"]) <= 8.0
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
