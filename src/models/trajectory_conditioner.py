"""Deterministic 20 Hz joint-trajectory time parameterization."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

ALGORITHM = "bounded-quintic-time-parameterizer"
TOPPRA_ALGORITHM = "toppra-spline-time-parameterizer"
TOPPRA_ALGORITHM_VERSION = 1
TOPPRA_CONSTRAINT_MARGIN = 0.98
ALGORITHM_VERSION = 1
DEFAULT_DEVIATION_BOUND_RAD = math.radians(5.0)
DEFAULT_MAX_CONDITIONED_STEPS = 4096
_EPS = 1e-9


def trajectory_metrics(
    actions: np.ndarray,
    history: Sequence[np.ndarray],
    period_s: float,
) -> dict[str, Any]:
    """Return finite-difference metrics using the measured two-sample anchor."""
    joints = np.asarray(actions[:, :6], dtype=np.float64)
    samples = np.vstack((history[-2], history[-1], joints))
    velocities = np.diff(samples, axis=0) / period_s
    accelerations = np.diff(samples, n=2, axis=0) / (period_s * period_s)
    return {
        "position_min_rad": np.min(joints, axis=0).tolist(),
        "position_max_rad": np.max(joints, axis=0).tolist(),
        "max_abs_velocity_rad_s": np.max(np.abs(velocities[1:]), axis=0).tolist(),
        "max_abs_acceleration_rad_s2": np.max(np.abs(accelerations), axis=0).tolist(),
        "first_action_delta_rad": (joints[0] - history[-1]).tolist(),
        "first_action_velocity_rad_s": velocities[1].tolist(),
        "first_action_acceleration_rad_s2": accelerations[0].tolist(),
    }


def _dynamics_valid(
    actions: np.ndarray,
    history: Sequence[np.ndarray],
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    period_s: float,
) -> bool:
    metrics = trajectory_metrics(actions, history, period_s)
    velocity = np.asarray(metrics["max_abs_velocity_rad_s"])
    acceleration = np.asarray(metrics["max_abs_acceleration_rad_s2"])
    return bool(
        np.all(velocity <= velocity_limits + _EPS)
        and np.all(acceleration <= acceleration_limits + _EPS)
    )


def _quintic_segment(
    start: np.ndarray,
    initial_velocity: np.ndarray,
    target: np.ndarray,
    steps: int,
    period_s: float,
) -> np.ndarray:
    duration = steps * period_s
    scaled_velocity = initial_velocity * duration
    delta = target - start
    a3 = 10.0 * delta - 6.0 * scaled_velocity
    a4 = -15.0 * delta + 8.0 * scaled_velocity
    a5 = 6.0 * delta - 3.0 * scaled_velocity
    u = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
    return start + scaled_velocity * u + a3 * u**3 + a4 * u**4 + a5 * u**5


def condition_trajectory(
    raw_actions: np.ndarray,
    history: Sequence[np.ndarray],
    profile: Any,
    *,
    period_s: float,
    deviation_bound_rad: float = DEFAULT_DEVIATION_BOUND_RAD,
    max_conditioned_steps: int = DEFAULT_MAX_CONDITIONED_STEPS,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Preserve raw waypoints while adding bounded quintic transition samples.

    Every raw waypoint remains an exact conditioned sample. Extra samples run at
    the unchanged control cadence. A duplicate endpoint stops each segment so a
    later waypoint cannot make an earlier segment dynamically unsafe.
    """
    actions = np.asarray(raw_actions, dtype=np.float64)
    measured = [np.asarray(value, dtype=np.float64) for value in history[-2:]]
    if len(measured) != 2 or any(value.shape != (6,) for value in measured):
        raise ValueError("conditioning requires measured two-sample joint history")
    velocity_limits = np.asarray(profile.velocity_limits, dtype=np.float64)
    acceleration_limits = np.asarray(profile.acceleration_limits, dtype=np.float64)
    base_diagnostics: dict[str, Any] = {
        "algorithm": ALGORITHM,
        "version": ALGORITHM_VERSION,
        "control_hz": 1.0 / period_s,
        "deviation_bound_rad": deviation_bound_rad,
        "raw_steps": int(actions.shape[0]),
        "raw_metrics": trajectory_metrics(actions, measured, period_s),
    }
    measured_velocity = np.abs(measured[-1] - measured[-2]) / period_s
    if np.any(measured_velocity > velocity_limits + _EPS):
        raise ValueError("measured joint history exceeds selected velocity limits")
    if _dynamics_valid(
        actions, measured, velocity_limits, acceleration_limits, period_s
    ):
        base_diagnostics.update(
            conditioned=False,
            conditioned_steps=int(actions.shape[0]),
            conditioned_duration_s=float(actions.shape[0] * period_s),
            max_path_deviation_rad=0.0,
            max_path_deviation_rad_per_joint=[0.0] * 6,
            max_target_delta_rad_per_joint=[0.0] * 6,
            source_waypoint_indices=list(range(actions.shape[0])),
            conditioned_metrics=base_diagnostics["raw_metrics"],
        )
        return actions.copy(), base_diagnostics

    conditioned: list[np.ndarray] = []
    source_indices: list[int] = []
    target_deltas: list[np.ndarray] = []
    path_deviations: list[np.ndarray] = []
    previous_previous, previous = measured
    previous_gripper = float(actions[0, 6])
    joint_min = np.asarray(profile.joint_min, dtype=np.float64)
    joint_max = np.asarray(profile.joint_max, dtype=np.float64)

    for source_index, raw_action in enumerate(actions):
        target = raw_action[:6]
        initial_velocity = (previous - previous_previous) / period_s
        accepted: np.ndarray | None = None
        for steps in range(1, 513):
            joints = _quintic_segment(
                previous, initial_velocity, target, steps, period_s
            )
            joints = np.vstack((joints, target))
            lows = np.minimum(previous, target)
            highs = np.maximum(previous, target)
            deviation = np.maximum(np.maximum(lows - joints, joints - highs), 0.0)
            if np.any(deviation > deviation_bound_rad + _EPS):
                continue
            if np.any(joints < joint_min - _EPS) or np.any(joints > joint_max + _EPS):
                continue
            candidate_actions = np.column_stack((joints, np.zeros(joints.shape[0])))
            if _dynamics_valid(
                candidate_actions,
                (previous_previous, previous),
                velocity_limits,
                acceleration_limits,
                period_s,
            ):
                accepted = joints
                break
        if accepted is None:
            raise ValueError(
                f"conditioning infeasible at raw waypoint {source_index} within "
                f"{deviation_bound_rad:.6f} rad path deviation"
            )
        if len(conditioned) + accepted.shape[0] > max_conditioned_steps:
            raise ValueError(
                "conditioning infeasible within maximum conditioned horizon "
                f"of {max_conditioned_steps} steps"
            )
        gripper = np.linspace(
            previous_gripper,
            float(raw_action[6]),
            accepted.shape[0] + 1,
            dtype=np.float64,
        )[1:]
        segment = np.column_stack((accepted, gripper))
        conditioned.extend(segment)
        source_indices.extend([source_index] * accepted.shape[0])
        target_deltas.extend(np.abs(accepted - target))
        lows = np.minimum(previous, target)
        highs = np.maximum(previous, target)
        path_deviations.extend(
            np.maximum(np.maximum(lows - accepted, accepted - highs), 0.0)
        )
        previous_previous, previous = accepted[-2], accepted[-1]
        previous_gripper = float(raw_action[6])

    result = np.asarray(conditioned, dtype=np.float64)
    path_delta = np.max(np.asarray(path_deviations), axis=0)
    target_delta = np.max(np.asarray(target_deltas), axis=0)
    base_diagnostics.update(
        conditioned=True,
        conditioned_steps=int(result.shape[0]),
        conditioned_duration_s=float(result.shape[0] * period_s),
        max_path_deviation_rad=float(np.max(path_delta)),
        max_path_deviation_rad_per_joint=path_delta.tolist(),
        max_target_delta_rad_per_joint=target_delta.tolist(),
        source_waypoint_indices=source_indices,
        conditioned_metrics=trajectory_metrics(result, measured, period_s),
    )
    return result, base_diagnostics


def _nearest_path_delta(
    position: np.ndarray, waypoints: np.ndarray
) -> tuple[np.ndarray, int]:
    best_distance = math.inf
    best_delta = np.zeros(6, dtype=np.float64)
    best_index = 0
    for index, (start, end) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        direction = end - start
        denominator = float(direction @ direction)
        progress = (
            0.0
            if denominator == 0.0
            else float(np.clip((position - start) @ direction / denominator, 0.0, 1.0))
        )
        delta = position - (start + progress * direction)
        distance = float(delta @ delta)
        if distance < best_distance:
            best_distance = distance
            best_delta = delta
            best_index = index
    return best_delta, best_index


def condition_trajectory_toppra(
    raw_actions: np.ndarray,
    history: Sequence[np.ndarray],
    profile: Any,
    *,
    period_s: float,
    deviation_bound_rad: float = DEFAULT_DEVIATION_BOUND_RAD,
    max_conditioned_steps: int = DEFAULT_MAX_CONDITIONED_STEPS,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Spline and time-parameterize a chunk without stopping at each waypoint."""
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint

    actions = np.asarray(raw_actions, dtype=np.float64)
    measured = [np.asarray(value, dtype=np.float64) for value in history[-2:]]
    if len(measured) != 2 or any(value.shape != (6,) for value in measured):
        raise ValueError("conditioning requires measured two-sample joint history")
    velocity_limits = np.asarray(profile.velocity_limits, dtype=np.float64)
    acceleration_limits = np.asarray(profile.acceleration_limits, dtype=np.float64)
    measured_velocity = np.abs(measured[-1] - measured[-2]) / period_s
    if np.any(measured_velocity > velocity_limits + _EPS):
        raise ValueError("measured joint history exceeds selected velocity limits")

    base_diagnostics: dict[str, Any] = {
        "algorithm": TOPPRA_ALGORITHM,
        "version": TOPPRA_ALGORITHM_VERSION,
        "control_hz": 1.0 / period_s,
        "constraint_margin": TOPPRA_CONSTRAINT_MARGIN,
        "deviation_bound_rad": deviation_bound_rad,
        "raw_steps": int(actions.shape[0]),
        "raw_metrics": trajectory_metrics(actions, measured, period_s),
    }
    if _dynamics_valid(
        actions, measured, velocity_limits, acceleration_limits, period_s
    ):
        base_diagnostics.update(
            conditioned=False,
            conditioned_steps=int(actions.shape[0]),
            conditioned_duration_s=float(actions.shape[0] * period_s),
            geometric_duration_s=float(actions.shape[0] * period_s),
            max_path_deviation_rad=0.0,
            max_path_deviation_rad_per_joint=[0.0] * 6,
            max_target_delta_rad_per_joint=[0.0] * 6,
            source_waypoint_indices=list(range(actions.shape[0])),
            conditioned_metrics=base_diagnostics["raw_metrics"],
        )
        return actions.copy(), base_diagnostics

    anchor_raw = np.repeat(actions[:1], 2, axis=0)
    anchor, _ = condition_trajectory(
        anchor_raw,
        measured,
        profile,
        period_s=period_s,
        deviation_bound_rad=deviation_bound_rad,
        max_conditioned_steps=max_conditioned_steps,
    )
    waypoints = actions[:, :6]
    unique = np.r_[True, np.linalg.norm(np.diff(waypoints, axis=0), axis=1) > _EPS]
    waypoints = waypoints[unique]
    geometric_duration = float(anchor.shape[0] * period_s)
    if waypoints.shape[0] < 2:
        result = anchor
    else:
        path_grid = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)),
        ]
        path_grid /= path_grid[-1]
        path = ta.SplineInterpolator(path_grid, waypoints)
        velocity_bounds = np.column_stack((-velocity_limits, velocity_limits))
        acceleration_bounds = np.column_stack(
            (-acceleration_limits, acceleration_limits)
        )
        instance = algo.TOPPRA(
            [
                constraint.JointVelocityConstraint(
                    velocity_bounds * TOPPRA_CONSTRAINT_MARGIN
                ),
                constraint.JointAccelerationConstraint(
                    acceleration_bounds * TOPPRA_CONSTRAINT_MARGIN
                ),
            ],
            path,
            solver_wrapper="seidel",
        )
        trajectory: Any = instance.compute_trajectory(0.0, 0.0)
        if trajectory is None or not math.isfinite(float(trajectory.duration)):
            raise ValueError("TOPPRA could not parameterize the raw joint path")
        path_duration = float(trajectory.duration)
        path_samples = int(math.ceil(path_duration / period_s))
        sample_times = np.minimum(
            np.arange(1, path_samples + 1, dtype=np.float64) * period_s,
            path_duration,
        )
        joints = np.asarray(trajectory(sample_times), dtype=np.float64)
        gripper = np.interp(
            sample_times,
            np.linspace(0.0, path_duration, actions.shape[0]),
            actions[:, 6],
        )
        result = np.vstack((anchor, np.column_stack((joints, gripper))))
        geometric_duration += path_duration
    sample_count = int(result.shape[0])
    if sample_count > max_conditioned_steps:
        raise ValueError(
            "conditioning infeasible within maximum conditioned horizon "
            f"of {max_conditioned_steps} steps"
        )
    joint_min = np.asarray(profile.joint_min, dtype=np.float64)
    joint_max = np.asarray(profile.joint_max, dtype=np.float64)
    if np.any(result[:, :6] < joint_min - _EPS) or np.any(
        result[:, :6] > joint_max + _EPS
    ):
        raise ValueError("TOPPRA spline exceeds selected joint position limits")
    if not _dynamics_valid(
        result, measured, velocity_limits, acceleration_limits, period_s
    ):
        raise ValueError("TOPPRA output exceeds discrete 20 Hz dynamic limits")

    path_deltas: list[np.ndarray] = []
    source_indices: list[int] = []
    target_deltas: list[np.ndarray] = []
    raw_waypoints = np.vstack((measured[-1], actions[:, :6]))
    for position in result[:, :6]:
        delta, segment_index = _nearest_path_delta(position, raw_waypoints)
        path_deltas.append(np.abs(delta))
        source_index = min(segment_index, actions.shape[0] - 1)
        source_indices.append(source_index)
        target_deltas.append(np.abs(position - actions[source_index, :6]))
    path_delta = np.max(np.asarray(path_deltas), axis=0)
    if np.any(path_delta > deviation_bound_rad + _EPS):
        raise ValueError(
            "TOPPRA spline exceeds path deviation bound: "
            f"{float(np.max(path_delta)):.6f} > {deviation_bound_rad:.6f} rad"
        )
    target_delta = np.max(np.asarray(target_deltas), axis=0)
    base_diagnostics.update(
        conditioned=True,
        conditioned_steps=sample_count,
        conditioned_duration_s=float(sample_count * period_s),
        geometric_duration_s=geometric_duration,
        max_path_deviation_rad=float(np.max(path_delta)),
        max_path_deviation_rad_per_joint=path_delta.tolist(),
        max_target_delta_rad_per_joint=target_delta.tolist(),
        source_waypoint_indices=source_indices,
        conditioned_metrics=trajectory_metrics(result, measured, period_s),
    )
    return result, base_diagnostics
