"""Experimental constrained HSMM challenger for session state labeling.

The model is intentionally separate from maneuver and crash admission. It
learns weak emission statistics from the current recording, decodes explicit
state durations with a beam, and reports provisional state labels. It is not a
calibrated classifier or a replacement for the heuristic event pipeline.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .analysis import wrap
from .types import Event, FloatArray, Segment


@dataclass(frozen=True, slots=True)
class State:
    """One flattened HSMM state and its duration prior."""

    name: str
    group: int
    side: int = 0
    maneuver: str = ""
    phase: str = ""
    mean_s: float = 30.0
    sigma: float = 0.65
    minimum_s: float = 1.0
    maximum_s: float = 240.0

    @property
    def parent(self) -> str:
        """Return the hierarchical parent label."""
        return "Foiling" if self.group >= 2 else self.name

    @property
    def hierarchy(self) -> str:
        """Return a slash-separated hierarchical state label."""
        return "/".join(
            [self.parent]
            + [value for value in (self.maneuver, self.phase) if value]
        )


@dataclass(frozen=True, slots=True)
class HSMMModel:
    """Fitted weak-label emission and duration parameters."""

    center: FloatArray
    scale: FloatArray
    means: FloatArray
    variances: FloatArray
    durations: dict[int, tuple[float, float, float, float]]


DEFAULT_DURATIONS: dict[int, tuple[float, float, float, float]] = {
    0: (40.0, 0.85, 1.0, 360.0),
    1: (12.0, 0.8, 1.0, 120.0),
    2: (45.0, 0.8, 2.0, 360.0),
    3: (45.0, 0.65, 12.0, 180.0),
    4: (5.0, 0.65, 1.0, 24.0),
    5: (12.0, 0.55, 2.0, 50.0),
    6: (10.0, 0.55, 1.0, 40.0),
}


def states_and_edges(
    durations: dict[int, tuple[float, float, float, float]] | None = None,
) -> tuple[list[State], dict[int, dict[int, float]]]:
    """Build the constrained 19-state topology and log transition weights."""
    priors = durations or DEFAULT_DURATIONS
    states = [
        State("Stopped", 0, mean_s=priors[0][0], sigma=priors[0][1], maximum_s=360),
        State("Taxiing", 1, mean_s=priors[1][0], sigma=priors[1][1], maximum_s=120),
        State("Unknown", -1, mean_s=8, sigma=1, maximum_s=360),
    ]
    for side in (1, -1):
        for phase, group in (("Reach", 2), ("Surf", 3)):
            mean, sigma, minimum, maximum = priors[group]
            states.append(
                State(
                    f"{phase}_{side:+d}", group, side, phase=phase,
                    mean_s=mean, sigma=sigma, minimum_s=minimum, maximum_s=maximum,
                )
            )
        for maneuver in ("Gybe", "Tack"):
            for phase, group in (("Setup", 4), ("Core", 5), ("Exit", 6)):
                mean, sigma, minimum, maximum = priors[group]
                states.append(
                    State(
                        f"{maneuver}_{side:+d}_{phase}", group, side,
                        maneuver, phase, mean, sigma, minimum, maximum,
                    )
                )
    names = {state.name: index for index, state in enumerate(states)}
    edges: dict[int, dict[int, float]] = {index: {} for index in range(len(states))}

    def add(source: int, target: str, weight: float) -> None:
        edges[source][names[target]] = weight

    for index, state in enumerate(states):
        if state.group in (-1, 0, 1):
            for target, candidate in enumerate(states):
                if target != index and (
                    candidate.group in (-1, 0, 1) or candidate.phase in ("Reach", "Surf")
                ):
                    edges[index][target] = 0.2 if candidate.phase == "Surf" else 1.0
            edges[index][index] = 0.1
        elif state.phase in ("Reach", "Surf"):
            other = "Surf" if state.phase == "Reach" else "Reach"
            add(index, f"{other}_{state.side:+d}", 2.0)
            add(index, f"Gybe_{state.side:+d}_Setup", 1.5)
            add(index, f"Tack_{state.side:+d}_Setup", 0.6)
            for target, weight in (("Taxiing", 0.6), ("Stopped", 0.1), ("Unknown", 0.03)):
                add(index, target, weight)
            edges[index][index] = 0.03
        elif state.phase == "Setup":
            add(index, f"{state.maneuver}_{state.side:+d}_Core", 4.0)
            for target, weight in (
                (f"Reach_{state.side:+d}", 0.4),
                (f"Surf_{state.side:+d}", 0.4),
                ("Taxiing", 0.4),
                ("Stopped", 0.1),
                ("Unknown", 0.03),
            ):
                add(index, target, weight)
        elif state.phase == "Core":
            add(index, f"{state.maneuver}_{state.side:+d}_Exit", 4.0)
            for target, weight in (
                (f"Reach_{state.side:+d}", 0.2),
                (f"Surf_{state.side:+d}", 0.2),
                ("Taxiing", 0.5),
                ("Stopped", 0.1),
                ("Unknown", 0.03),
            ):
                add(index, target, weight)
        elif state.phase == "Exit":
            add(index, f"Reach_{-state.side:+d}", 4.0)
            add(index, f"Surf_{-state.side:+d}", 2.0)
            for target, weight in (("Taxiing", 0.5), ("Stopped", 0.1), ("Unknown", 0.03)):
                add(index, target, weight)
    for source, transitions in edges.items():
        total = sum(transitions.values())
        edges[source] = {target: math.log(weight / total) for target, weight in transitions.items()}
    return states, edges


def survival(age_s: float, state: State) -> float:
    """Return the truncated-lognormal duration survival probability."""
    if age_s < state.minimum_s:
        return 1.0
    if age_s >= state.maximum_s:
        return 1e-12
    mu = math.log(state.mean_s) - 0.5 * state.sigma**2

    def cdf(duration: float) -> float:
        return float(ndtr((math.log(max(duration, 1e-9)) - mu) / state.sigma))

    lower = cdf(state.minimum_s)
    upper = cdf(state.maximum_s)
    return max(1e-12, (upper - cdf(age_s)) / max(upper - lower, 1e-12))


def features(t: FloatArray, c: FloatArray, v: FloatArray, wind: float) -> FloatArray:
    """Build the seven causal features used by the weak-label model."""
    values = np.full((len(t), 7), np.nan)
    relative = wrap(c - np.deg2rad(wind))
    values[:, 0] = v
    values[:, 1] = abs(np.sin(relative))
    values[:, 2] = -np.cos(relative)
    dt = np.diff(t)
    rate = np.rad2deg(wrap(np.diff(c))) / dt
    rate[(dt > 8) | (v[:-1] < 2) | (v[1:] < 2)] = np.nan
    values[1:, 3] = abs(rate)
    values[1:, 4] = rate * np.sign(np.sin(relative[1:]))
    for index in range(len(t)):
        sample_ids = np.arange(np.searchsorted(t, t[index] - 12), index + 1)
        valid = sample_ids[np.isfinite(c[sample_ids]) & (v[sample_ids] >= 2)]
        if len(valid) >= 3 and np.max(np.diff(t[valid])) <= 8:
            increments = wrap(np.diff(c[valid]))
            total = np.sum(np.abs(increments), keepdims=False)
            values[index, 5] = abs(np.sum(increments, keepdims=False)) / max(total, 0.001)
        recent = v[sample_ids]
        recent = recent[np.isfinite(recent)]
        if len(recent):
            values[index, 6] = v[index] / max(1, np.median(recent))
    values[v < 2, 1:6] = np.nan
    return values


def weak_labels(
    t: FloatArray, segments: list[Segment], events: list[Event]
) -> np.ndarray:
    """Convert current heuristic output into explicit-duration training labels."""
    labels = np.full(len(t), -1, dtype=int)
    mapping = {"Stopped / drifting": 0, "Slow / recovery": 1, "Reaching": 2, "Surf": 3}
    for segment in segments:
        if segment.type in mapping:
            labels[segment.a : segment.b] = mapping[segment.type]
    for event in events:
        if event.core_a is None or event.core_b is None:
            continue
        labels[event.a : event.core_a] = 4
        labels[event.core_a : event.core_b] = 5
        labels[event.core_b : event.b] = 6
    return labels


def fit_model(values: FloatArray, labels: np.ndarray) -> HSMMModel:
    """Fit weak-label emission parameters with conservative variance floors."""
    valid = labels >= 0
    center = np.nanmean(values[valid], axis=0)
    scale = np.maximum(np.nanstd(values[valid], axis=0), 0.05)
    standardized = (values - center) / scale
    means = []
    variances = []
    for group in range(7):
        samples = standardized[valid & (labels == group)]
        if len(samples) < 3:
            means.append(np.zeros(values.shape[1]))
            variances.append(np.ones(values.shape[1]))
        else:
            means.append(np.nan_to_num(np.nanmean(samples, axis=0)))
            variances.append(np.maximum(np.nan_to_num(np.nanvar(samples, axis=0), nan=1), 0.15))
    return HSMMModel(
        center=center,
        scale=scale,
        means=np.asarray(means),
        variances=np.asarray(variances),
        durations=DEFAULT_DURATIONS.copy(),
    )


def decode(
    t: FloatArray,
    c: FloatArray,
    v: FloatArray,
    wind: float,
    model: HSMMModel,
    beam: int = 96,
) -> np.ndarray:
    """Decode a state path with explicit duration survival and beam pruning."""
    states, edges = states_and_edges(model.durations)
    values = (features(t, c, v, wind) - model.center) / model.scale
    emissions = np.full((len(t), len(states)), -25.0)
    for index, state in enumerate(states):
        if state.group < 0:
            continue
        group_mean = model.means[state.group]
        group_variance = model.variances[state.group]
        emissions[:, index] = 0.35 * np.nansum(
            -0.5 * (np.log(2 * np.pi * group_variance) + (values - group_mean) ** 2 / group_variance),
            axis=1,
        )
        if state.phase in ("Reach", "Surf", "Setup", "Exit"):
            side = -state.side if state.phase == "Exit" else state.side
            reliable = np.isfinite(c) & (v >= 2) & (abs(np.sin(wrap(c - np.deg2rad(wind)))) > 0.2)
            emissions[reliable & (np.sign(np.sin(wrap(c - np.deg2rad(wind)))) != side), index] = -1e6
    paths: list[tuple[float, int, float, list[int]]] = []
    initial = [index for index, state in enumerate(states) if state.group in (-1, 0, 1, 2, 3)]
    for index in initial:
        paths.append((float(emissions[0, index]), index, 0.0, [index]))
    for sample in range(1, len(t)):
        candidates: list[tuple[float, int, float, list[int]]] = []
        for score, state_index, age, path in paths:
            state = states[state_index]
            dt = float(t[sample] - t[sample - 1])
            stay_probability = survival(age, state)
            remaining = min(1.0, survival(age + dt, state) / stay_probability)
            if age + dt < state.maximum_s:
                candidates.append((score + math.log(max(remaining, 1e-12)) + emissions[sample, state_index], state_index, age + dt, path + [state_index]))
            for target, transition in edges[state_index].items():
                candidates.append((score + math.log(max(1 - remaining, 1e-12)) + transition + emissions[sample, target], target, 0.0, path + [target]))
        paths = sorted(candidates, key=lambda item: item[0], reverse=True)[:beam]
    return np.asarray(paths[0][3], dtype=int)


def state_records(t: FloatArray, labels: np.ndarray, states: list[State]) -> list[dict[str, object]]:
    """Convert a decoded state path into JSON-ready sample records."""
    return [
        {
            "time_s": float(time_s),
            "state": states[int(label)].name,
            "parent": states[int(label)].parent,
            "hierarchy": states[int(label)].hierarchy,
        }
        for time_s, label in zip(t, labels)
    ]