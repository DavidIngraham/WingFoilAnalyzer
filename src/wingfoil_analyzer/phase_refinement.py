"""Retrospective setup/core/exit refinement for detected maneuvers."""

import numpy as np

from .analysis import wrap
from .types import ArcMetrics, Event, FloatArray


def arc_metrics(
    t: FloatArray,
    c: FloatArray,
    v: FloatArray,
    xy: FloatArray,
    start: int,
    end: int,
) -> ArcMetrics:
    """Measure reliable path length, effective radius, and continuity."""
    if end <= start:
        return ArcMetrics(None, None, None)
    ids = np.arange(start, end + 1)
    dt = np.diff(t[ids])
    angles = wrap(np.diff(c[ids]))
    distances = np.linalg.norm(np.diff(xy[ids], axis=0), axis=1)
    good = (
        np.isfinite(angles)
        & np.isfinite(distances)
        & (v[ids[:-1]] >= 2)
        & (v[ids[1:]] >= 2)
        & (dt <= 8)
    )
    angles = np.where(np.abs(angles) < np.deg2rad(3), 0, angles)[good]
    dt = dt[good]
    if len(angles) < 2:
        return ArcMetrics(None, None, None)
    length = float(np.sum(distances[good], keepdims=False))
    total = float(np.sum(np.abs(angles), keepdims=False))
    net = float(np.sum(angles, keepdims=False))
    flowing = float(
        np.sum(dt[np.abs(angles) / dt >= np.deg2rad(3)], keepdims=False)
        / sum(dt)
    )
    continuity = (
        round(100 * (0.7 * abs(net) / total + 0.3 * flowing))
        if total > np.pi / 6
        else None
    )
    radius = (
        round(length / total, 1)
        if total >= np.pi / 4 and len(angles) >= 3
        else None
    )
    return ArcMetrics(round(length, 1), radius, continuity)


def refine_phases(
    t: FloatArray,
    c: FloatArray,
    v: FloatArray,
    xy: FloatArray,
    wind: float,
    events: list[Event],
) -> list[Event]:
    """Add retrospective setup/core/exit boundaries to gybe and tack events.

    The parent event interval and outcome remain unchanged. Core rotation is
    selected from sustained signed turn rate, with short gaps bridged.
    """
    rate = np.full(len(t), np.nan)
    dt = np.diff(t)
    rate[1:] = np.rad2deg(wrap(np.diff(c))) / dt
    rate[1:][(dt > 8) | (v[:-1] < 2) | (v[1:] < 2)] = np.nan
    for event in events:
        if event.type not in ("Gybe", "Tack"):
            continue
        valid = np.arange(event.a, event.b + 1)
        valid = valid[np.isfinite(c[valid]) & (v[valid] >= 2)]
        if len(valid) < 3:
            event.phase_status = "Insufficient data"
            continue
        direction = 1 if event.turn_direction == "Right / CW" else -1
        alpha = wrap(c[valid] - np.deg2rad(wind))
        target = np.pi if event.type == "Gybe" else 0
        anchor = int(valid[np.argmin(np.abs(wrap(alpha - target)))])
        active = np.flatnonzero(
            (np.arange(len(t)) > event.a)
            & (np.arange(len(t)) <= event.b)
            & np.isfinite(rate)
            & (direction * rate >= 3)
        )
        clusters: list[list[int]] = []
        for index in active:
            if (
                clusters
                and t[index - 1] - t[clusters[-1][-1]] <= 6
                and not np.any(
                    direction * rate[clusters[-1][-1] + 1 : index] < -5
                )
            ):
                clusters[-1].append(int(index))
            else:
                clusters.append([int(index)])
        if not clusters:
            event.phase_status = "No sustained core"
            continue
        cluster = min(
            clusters,
            key=lambda group: max(
                t[group[0] - 1] - t[anchor], t[anchor] - t[group[-1]], 0
            ),
        )
        core_a = max(event.a, cluster[0] - 1)
        core_b = min(event.b, cluster[-1])
        axis = min(core_b, max(core_a, anchor))
        core = arc_metrics(t, c, v, xy, core_a, core_b)
        entry = arc_metrics(t, c, v, xy, core_a, axis)
        exit_metrics = arc_metrics(t, c, v, xy, axis, core_b)
        event.core_a = core_a
        event.core_b = core_b
        event.axis_index = axis
        event.core_start_s = round(float(t[core_a] - t[0]), 1)
        event.core_end_s = round(float(t[core_b] - t[0]), 1)
        event.setup_duration_s = round(float(t[core_a] - t[event.a]), 1)
        event.core_duration_s = round(float(t[core_b] - t[core_a]), 1)
        event.exit_observation_s = round(float(t[event.b] - t[core_b]), 1)
        event.core_length_m = core.length_m
        event.core_radius_m = core.radius_m
        event.core_continuity = core.continuity
        event.entry_radius_m = entry.radius_m
        event.exit_radius_m = exit_metrics.radius_m
        event.phase_status = (
            "Refined" if core_a <= anchor <= core_b else "Uncertain"
        )
    return events