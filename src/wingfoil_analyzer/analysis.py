"""Course estimation, maneuver detection, and session segmentation."""

from typing import Any

import numpy as np

from .types import (
    Crash,
    Event,
    FloatArray,
    FoilOnset,
    IntArray,
    Segment,
    Session,
    SurfMetrics,
)


def wrap(a: Any) -> Any:
    """Normalize an angle or angle array to the interval [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def courses(t: FloatArray, xy: FloatArray, window: float) -> FloatArray:
    """Estimate course angles from trailing local linear position fits."""
    c = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        j = min(np.searchsorted(t, t[i] - window), i - 1)
        p = xy[j : i + 1]
        tt = t[j : i + 1]
        if not np.isfinite(p).all() or np.max(np.diff(tt)) > 8:
            continue
        q = tt - float(tt.mean())
        v = q @ (p - p.mean(0)) / (q @ q)
        if np.linalg.norm(v) > 1:
            c[i] = np.arctan2(v[0], v[1])
    return c


def events(
    t: FloatArray, c: FloatArray, v: FloatArray, wind: float, xy: FloatArray
) -> list[Event]:
    """Detect turn attempts and calculate their maneuver and exit metrics."""
    alpha = wrap(c - np.deg2rad(wind))
    out = []
    i = 1
    n = len(t)
    while i < n - 2:
        # Stable powered reach supplies a fixed entry heading, not a moving target.
        pre = np.flatnonzero(
            (t >= t[i] - 12) & (t <= t[i] - 2) & np.isfinite(c) & (v >= 3)
        )
        if len(pre) < 3 or t[pre[-1]] - t[pre[0]] < 6:
            i += 1
            continue
        base = np.angle(np.mean(np.exp(1j * c[pre])))
        entryalpha = wrap(base - np.deg2rad(wind))
        if (
            np.max(np.abs(wrap(c[pre] - float(base)))) > np.deg2rad(18)
            or abs(np.sin(entryalpha)) < 0.4
        ):
            i += 1
            continue
        if not np.isfinite(c[i]) or v[i] < 2 or abs(wrap(c[i] - base)) < np.deg2rad(25):
            i += 1
            continue
        if out and pre[0] <= out[-1].b:
            i += 1
            continue
        a = max(pre[-1], i - 1)
        # Include preceding setup excursions, up to six seconds, without pulling in a whole reach.
        for k in range(i - 1, pre[0] - 1, -1):
            if t[i] - t[k] > 6:
                break
            a = k
        direction = 1 if wrap(c[i] - base) > 0 else -1
        # Initial signed movement toward downwind distinguishes gybe from tack.
        kind = "Gybe" if direction * np.sin(entryalpha) > 0 else "Tack"
        startside = np.sign(np.sin(entryalpha))
        lastc = base
        rot = 0.0
        maxrot = 0.0
        comp = None
        opp = None
        low = None
        back = None
        commit = None
        reason = "timeout"
        b = i
        for j in range(i, n):
            b = j
            if np.isfinite(c[j]) and v[j] >= 2:
                rot += float(wrap(c[j] - lastc))
                lastc = c[j]
                maxrot = max(maxrot, direction * rot)
                if maxrot >= np.deg2rad(75) and commit is None:
                    commit = j
                opposite = (
                    np.sign(np.sin(alpha[j])) == -startside
                    and abs(np.sin(alpha[j])) > 0.5
                )
                if opposite and maxrot >= np.deg2rad(75):
                    if opp is None:
                        opp = j
                    if t[j] - t[opp] >= 2 and comp is None:
                        comp = opp
                else:
                    opp = None
                if abs(wrap(c[j] - base)) < np.deg2rad(18):
                    if back is None:
                        back = j
                else:
                    back = None
            if np.isfinite(v[j]) and v[j] < 1.5:
                if low is None:
                    low = j
            else:
                low = None
            if low is not None and t[j] - t[low] >= 5:
                reason = "stop"
                break
            if comp is not None and t[j] - t[comp] >= 10 and low is None:
                reason = "exit"
                break
            if comp is None and back is not None and t[j] - t[back] >= 6:
                reason = "abort"
                break
            if t[j] - t[a] >= 45:
                break
        if b == n - 1 and reason == "timeout":
            reason = "end of recording"
        accepted = maxrot >= np.deg2rad(75) or comp is not None
        if not accepted:
            i += 1
            continue
        sl = slice(a, b + 1)
        maxgap = float(np.max(np.diff(t[sl])))
        if reason == "stop":
            outcome = (
                "Probable crash after completion"
                if comp is not None
                else "Probable crash during turn"
            )
        elif comp is not None:
            exitv = v[comp : b + 1]
            outcome = (
                "Speed-retaining exit"
                if np.nanmin(exitv) >= 3.5
                else (
                    "Slow exit / recovered" if v[b] >= 3 else "Slow exit / unresolved"
                )
            )
        else:
            outcome = (
                "Aborted / returned" if reason == "abort" else "Unresolved attempt"
            )
        arcend = comp if comp is not None else (low if low is not None else b)
        ids = np.arange(a, arcend + 1)
        ids = ids[np.isfinite(c[ids]) & (v[ids] >= 2)]
        total = 0.0
        net = 0.0
        stall = 0.0
        observed = 0.0
        counter = 0.0
        arc = 0.0
        for left, right in zip(ids[:-1], ids[1:]):
            dt = t[right] - t[left]
            if dt > 8:
                continue
            da = float(wrap(c[right] - c[left]))
            da = 0.0 if abs(da) < np.deg2rad(3) else da
            total += abs(da)
            net += da
            counter += max(0, -direction * da)
            observed += dt
            if abs(da) / dt < np.deg2rad(3):
                stall += dt
            if np.isfinite(xy[[left, right]]).all():
                arc += float(np.linalg.norm(xy[right] - xy[left]))
        directional = 100 * abs(net) / total if total else 0.0
        temporal = 100 * (1 - stall / observed) if observed else 0.0
        continuity = (
            round(float(0.7 * directional + 0.3 * temporal))
            if len(ids) >= 4 and total > np.deg2rad(30)
            else None
        )
        radius = (
            round(arc / total, 1) if len(ids) >= 4 and total >= np.deg2rad(45) else None
        )
        quality = "Low" if maxgap > 5 or len(ids) < 5 or comp is None else "Moderate"
        out.append(
            Event(
                a=a,
                b=b,
                type=kind,
                outcome=outcome,
                turn_direction="Right / CW" if direction > 0 else "Left / CCW",
                continuity=continuity,
                score_scope="Complete" if comp is not None else "Partial",
                directional_score=round(directional),
                pause_s=round(float(stall), 1),
                counterturn_deg=round(np.rad2deg(counter), 1),
                setup_s=round(t[commit] - t[a], 1) if commit is not None else None,
                radius_m=radius,
                rotation_deg=round(np.rad2deg(maxrot), 1),
                completion_s=round(t[comp] - t[0], 1) if comp is not None else None,
                completion_to_stop_s=round(t[low] - t[comp], 1)
                if comp is not None and low is not None
                else None,
                min_speed_mps=round(float(np.nanmin(v[sl])), 2),
                confidence=quality,
            )
        )
        i = b + 1
    return out


def recover_transitions(
    t: FloatArray,
    c: FloatArray,
    v: FloatArray,
    wind: float,
    xy: FloatArray,
    ev: list[Event],
) -> list[Event]:
    """Detect sustained side changes without a steady-entry requirement (surf exits)."""
    alpha = wrap(c - np.deg2rad(wind))
    side = np.where(
        (v >= 2.5) & np.isfinite(c) & (np.abs(np.sin(alpha)) > 0.6),
        np.sign(np.sin(alpha)),
        0,
    )
    current = 0
    last = None
    pending = 0
    start = None
    added = []
    for i in range(len(t)):
        if not side[i]:
            pending = 0
            start = None
            continue
        if side[i] == current:
            last = i
            pending = 0
            start = None
            continue
        if side[i] != pending:
            pending = side[i]
            start = i
        if start is None:
            continue
        if t[i] - t[start] < 3:
            continue
        if current and last is not None and t[start] - t[last] <= 25:
            ids = np.arange(last, start + 1)
            ids = ids[np.isfinite(c[ids]) & (v[ids] >= 2)]
            covered = any(
                e.a <= start <= e.b and e.outcome != "Unresolved attempt"
                for e in ev + added
            )
            if len(ids) >= 2 and not covered and np.max(np.diff(t[ids])) <= 8:
                near = ids[np.argmin(np.abs(np.sin(alpha[ids])))]
                kind = "Gybe" if np.cos(alpha[near]) < 0 else "Tack"
                a = max(0, int(np.searchsorted(t, t[last] - 6)))
                comp = start
                assert comp is not None
                b = min(len(t) - 1, int(np.searchsorted(t, t[comp] + 10)))
                low = None
                for j in range(comp, min(len(t), b + 8)):
                    if v[j] < 1.5:
                        if low is None:
                            low = j
                    else:
                        low = None
                    if low is not None and t[j] - t[low] >= 5:
                        b = j
                        break
                    if j >= b and low is None:
                        break
                # Preserve neighboring established events and exclude any overlapping fallback.
                preceding = [
                    e.b
                    for e in ev + added
                    if e.outcome != "Unresolved attempt" and e.b < comp
                ]
                if preceding:
                    a = max(a, max(preceding))
                following = [
                    e.a
                    for e in ev + added
                    if e.outcome != "Unresolved attempt" and e.a > comp
                ]
                if following:
                    b = min(b, min(following))
                if a < comp < b:
                    outcome = (
                        "Probable crash after completion"
                        if low is not None and t[b] - t[low] >= 5
                        else (
                            "Speed-retaining exit"
                            if np.nanmin(v[comp : b + 1]) >= 3.5
                            else (
                                "Slow exit / recovered"
                                if v[b] >= 3
                                else "Slow exit / unresolved"
                            )
                        )
                    )
                    kk = np.arange(a, comp + 1)
                    kk = kk[np.isfinite(c[kk]) & (v[kk] >= 2)]
                    da = wrap(np.diff(c[kk]))
                    dt = np.diff(t[kk])
                    good = dt <= 8
                    da = np.where(np.abs(da) < np.deg2rad(3), 0, da)
                    da = da[good]
                    dt = dt[good]
                    total = float(np.sum(np.abs(da), keepdims=False))
                    net = float(np.sum(da, keepdims=False))
                    direction = 1 if net >= 0 else -1
                    pause = float(
                        np.sum(dt[np.abs(da) / dt < np.deg2rad(3)], keepdims=False)
                    )
                    duration = float(np.sum(dt, keepdims=False))
                    directional = 100 * abs(net) / total if total else 0
                    score = (
                        round(0.7 * directional + 0.3 * 100 * (1 - pause / duration))
                        if len(kk) >= 4 and duration and total > 0.52
                        else None
                    )
                    arc = float(
                        np.sum(
                            np.linalg.norm(np.diff(xy[kk], axis=0), axis=1)[good],
                            keepdims=False,
                        )
                    )
                    added.append(
                        Event(
                            a=a,
                            b=b,
                            type=kind,
                            outcome=outcome,
                            turn_direction="Right / CW"
                            if direction > 0
                            else "Left / CCW",
                            continuity=score,
                            score_scope="Complete",
                            directional_score=round(directional),
                            pause_s=round(pause, 1),
                            counterturn_deg=round(
                                float(
                                    np.rad2deg(
                                        np.sum(
                                            np.maximum(0, -direction * da),
                                            keepdims=False,
                                        )
                                    )
                                ),
                                1,
                            ),
                            setup_s=None,
                            radius_m=round(arc / total, 1)
                            if total >= np.pi / 4 and len(kk) >= 4
                            else None,
                            rotation_deg=round(abs(np.rad2deg(net)), 1),
                            completion_s=round(t[comp] - t[0], 1),
                            completion_to_stop_s=round(t[low] - t[comp], 1)
                            if low is not None
                            else None,
                            min_speed_mps=round(float(np.nanmin(v[a : b + 1])), 2),
                            confidence="Low",
                            detection_source="Sustained side change / variable approach",
                        )
                    )
        current = side[i]
        last = i
        pending = 0
        start = None
    kept = [
        e
        for e in ev
        if not (
            e.outcome == "Unresolved attempt"
            and any(e.a < q.b and q.a < e.b for q in added)
        )
    ]
    for e in kept:
        e.detection_source = "Stable approach"
    return sorted(kept + added, key=lambda e: e.a)


def surf_metrics(
    t: FloatArray, c: FloatArray, xy: FloatArray, ids: IntArray, wind: float
) -> SurfMetrics:
    """Calculate oscillation, heading, and crosswind metrics for a sample window."""
    # Hysteretic heading extrema suppress small GPS fluctuations.
    h = np.rad2deg(np.unwrap(c[ids]))
    ext = [(ids[0], h[0])]
    direction = 0
    for idx, val in zip(ids[1:], h[1:]):
        if direction == 0:
            if abs(val - ext[0][1]) >= 18:
                direction = np.sign(val - ext[0][1])
                ext.append((idx, val))
        elif direction * (val - ext[-1][1]) >= 0:
            ext[-1] = (idx, val)
        elif direction * (val - ext[-1][1]) <= -18:
            direction = -direction
            ext.append((idx, val))
    delta = xy[ids[-1]] - xy[ids[0]]
    bearing = np.arctan2(delta[0], delta[1])
    alpha = wrap(bearing - np.deg2rad(wind))
    dt = np.diff(t[ids])
    side = np.sin(c[ids[:-1]] - np.deg2rad(wind)) > 0
    fraction = float(np.sum(dt * side, keepdims=False) / sum(dt))
    same = max(fraction, 1 - fraction)
    return SurfMetrics(
        reversals=max(0, len(ext) - 2),
        heading_range_deg=float(np.ptp(h)),
        crosswind_fraction=float(abs(np.sin(alpha))),
        same_side_fraction=same,
        mean_course_deg=float(np.rad2deg(bearing) % 360),
        ext=ext,
    )


def add_surf(
    t: FloatArray,
    c: FloatArray,
    v: FloatArray,
    wind: float,
    xy: FloatArray,
    ev: list[Event],
) -> list[Event]:
    """Add sustained crosswind carving segments without masking protected events."""
    # Protect developed gybes/crashes. Only unresolved candidates may be superseded.
    allowed = np.isfinite(c) & np.isfinite(xy).all(axis=1) & (v >= 2)
    for e in ev:
        if e.outcome != "Unresolved attempt":
            allowed[e.a : e.b + 1] = False
    seeds = []
    for b in range(len(t)):
        a = np.searchsorted(t, t[b] - 36)
        ids = np.arange(a, b + 1)
        if (
            len(ids) < 8
            or t[b] - t[a] < 24
            or not allowed[ids].all()
            or np.max(np.diff(t[ids])) > 5
        ):
            continue
        m = surf_metrics(t, c, xy, ids, wind)
        # Sustained oscillation while net travel stays within 40 degrees of crosswind.
        if (
            m.reversals < 3
            or m.heading_range_deg < 40
            or m.heading_range_deg > 140
            or m.crosswind_fraction < 0.766
            or m.same_side_fraction < 0.9
        ):
            continue
        left = m.ext[0][0]
        right = m.ext[-1][0]
        if t[right] - t[left] < 18:
            continue
        if seeds and left <= seeds[-1][1]:
            seeds[-1][1] = max(right, seeds[-1][1])
        else:
            seeds.append([left, right])
    # Remove only unresolved attempts predominantly covered by sustained surf evidence.
    kept = []
    for e in ev:
        overlap = sum(
            max(0, min(t[e.b], t[b]) - max(t[e.a], t[a])) for a, b in seeds
        )
        if e.outcome == "Unresolved attempt" and overlap >= 0.5 * (
            t[e.b] - t[e.a]
        ):
            continue
        kept.append(e)
    # Split remaining surf at protected event boundaries, then recheck each fragment.
    mask = np.zeros(len(t) - 1, dtype=bool)
    for a, b in seeds:
        mask[a:b] = True
    for e in kept:
        mask[e.a : e.b] = False
    runs = []
    a = 0
    while a < len(mask):
        if not mask[a]:
            a += 1
            continue
        b = a + 1
        while b < len(mask) and mask[b]:
            b += 1
        runs.append((a, b))
        a = b
    for a, b in runs:
        ids = np.arange(a, b + 1)
        if t[b] - t[a] < 18 or not allowed[ids].all():
            continue
        m = surf_metrics(t, c, xy, ids, wind)
        if (
            m.reversals < 3
            or m.crosswind_fraction < 0.766
            or m.same_side_fraction < 0.9
        ):
            continue
        kept.append(
            Event(
                a=a,
                b=b,
                type="Surf",
                outcome="Sustained crosswind carving",
                turn_direction="Alternating",
                continuity=None,
                score_scope="",
                directional_score=None,
                pause_s=None,
                counterturn_deg=None,
                setup_s=None,
                radius_m=None,
                rotation_deg=None,
                completion_s=None,
                completion_to_stop_s=None,
                min_speed_mps=round(float(np.min(v[ids])), 2),
                confidence="Moderate",
                carve_reversals=m.reversals,
                heading_range_deg=round(m.heading_range_deg, 1),
                mean_course_deg=round(m.mean_course_deg, 1),
                crosswind_fraction=round(m.crosswind_fraction, 3),
            )
        )
    kept.sort(key=lambda e: e.a)
    assert all(a.b <= b.a for a, b in zip(kept, kept[1:]))
    return kept


def crashes(
    t: FloatArray, v: FloatArray, xy: FloatArray, ev: list[Event]
) -> list[Crash]:
    """Discrete probable falls: rapid powered-to-low-speed transition, 5 s persistence."""
    out = []
    i = 0
    while i < len(t):
        if not np.isfinite(v[i]) or v[i] >= 1.5:
            i += 1
            continue
        a = i
        b = i
        while (
            b + 1 < len(t)
            and np.isfinite(v[b + 1])
            and v[b + 1] < 1.5
            and t[b + 1] - t[b] <= 8
        ):
            b += 1
        if t[b] - t[a] >= 5:
            prior = np.flatnonzero(
                (t >= t[a] - 12) & (t < t[a]) & np.isfinite(v) & (v >= 3.5)
            )
            if (
                len(prior) >= 2
                and t[prior[-1]] - t[prior[0]] >= 2
                and np.isfinite(xy[a]).all()
            ):
                context = next((e for e in ev if e.a <= a <= e.b), None)
                phase = "Outside detected maneuver"
                if context:
                    if context.type == "Surf":
                        phase = "During surf"
                    elif (
                        context.completion_s is not None
                        and t[a] - t[0] >= context.completion_s
                    ):
                        phase = "After turn completion"
                    else:
                        phase = "During turn"
                gap = float(np.max(np.diff(t[prior[-1] : b + 1])))
                out.append(
                    Crash(
                        id="C" + str(len(out) + 1),
                        type="Probable crash",
                        time_s=round(t[a] - t[0], 1),
                        sample_index=a,
                        low_speed_duration_s=round(t[b] - t[a], 1),
                        speed_before_mps=round(float(v[prior[-1]]), 2),
                        speed_after_mps=round(float(v[a]), 2),
                        context=phase,
                        confidence="Low" if gap > 5 else "Moderate",
                        timing_uncertainty_s=round(float(t[a] - t[max(0, a - 1)]), 1),
                    )
                )
        i = b + 1
    return out


def foil_onsets(
    t: FloatArray, v: FloatArray, xy: FloatArray, ev: list[Event]
) -> list[FoilOnset]:
    """Detect low-speed-to-powered transitions sustained for at least two seconds."""
    out = []
    i = 1
    while i < len(t):
        if (
            not np.isfinite(v[i])
            or v[i] < 3.5
            or not np.isfinite(xy[i]).all()
            or np.isfinite(v[i - 1])
            and v[i - 1] >= 3.5
        ):
            i += 1
            continue
        b = i
        while (
            b + 1 < len(t)
            and np.isfinite(v[b + 1])
            and v[b + 1] >= 3.5
            and t[b + 1] - t[b] <= 8
        ):
            b += 1
        if t[b] - t[i] < 2:
            i += 1
            continue
        context = next(
            (
                "During detected maneuver"
                for event in ev
                if event.a <= i <= event.b
            ),
            "Outside detected maneuver",
        )
        max_gap = float(np.max(np.diff(t[i : b + 1])))
        out.append(
            FoilOnset(
                id="F" + str(len(out) + 1),
                type="Onto foil",
                time_s=round(t[i] - t[0], 1),
                sample_index=i,
                speed_mps=round(float(v[i]), 2),
                speed_before_mps=round(float(v[i - 1]), 2),
                powered_duration_s=round(float(t[b] - t[i]), 1),
                context=context,
                confidence="Low" if max_gap > 5 else "Moderate",
            )
        )
        i = b + 1
    return out


COLORS = {
    "Reaching": "#2878b5",
    "Surf": "#e14c91",
    "Gybe": "#119c83",
    "Tack": "#9163c5",
    "Slow / recovery": "#e5a238",
    "Stopped / drifting": "#8b94a3",
    "Unknown / gap": "#d1d5db",
}


def segment(
    t: FloatArray,
    xy: FloatArray,
    v: FloatArray,
    ev: list[Event],
    session: Session,
    ) -> list[Segment]:
    """Label contiguous session intervals and preserve complete elapsed-time coverage."""
    labels = []
    for i in range(len(t) - 1):
        vv = np.nanmean(v[i : i + 2]) if np.isfinite(v[i : i + 2]).any() else np.nan
        labels.append(
            "Unknown / gap"
            if not np.isfinite(xy[i : i + 2]).all()
            or t[i + 1] - t[i] > 8
            or not np.isfinite(vv)
            else (
                "Reaching"
                if vv >= 3
                else ("Slow / recovery" if vv >= 1.5 else "Stopped / drifting")
            )
        )
    # Merge short residual speed-state islands to avoid threshold chatter (offline labeling only).
    for _ in range(2):
        runs = []
        a = 0
        for b in range(1, len(labels) + 1):
            if b == len(labels) or labels[b] != labels[a]:
                runs.append((a, b, labels[a]))
                a = b
        for k, (a, b, lab) in enumerate(runs):
            if (
                0 < k < len(runs) - 1
                and t[b] - t[a] < 6
                and lab != "Unknown / gap"
                and runs[k - 1][2] == runs[k + 1][2]
            ):
                labels[a:b] = [runs[k - 1][2]] * (b - a)
    for k, e in enumerate(ev):
        labels[e.a : e.b] = [f"E{k}"] * (e.b - e.a)
    seg = []
    a = 0
    for b in range(1, len(labels) + 1):
        if b < len(labels) and labels[b] == labels[a]:
            continue
        lab = labels[a]
        event = ev[int(lab[1:])] if lab.startswith("E") else None
        kind = event.type if event is not None else lab
        d = np.linalg.norm(np.diff(xy[a : b + 1], axis=0), axis=1)
        d[np.diff(t[a : b + 1]) > 8] = np.nan
        dist = float(np.nansum(d, keepdims=False)) if np.isfinite(d).any() else None
        s = Segment(
            id=len(seg) + 1,
            type=kind,
            start_s=round(t[a] - t[0], 3),
            end_s=round(t[b] - t[0], 3),
            duration_s=round(t[b] - t[a], 3),
            reach_length_m=round(dist, 1)
            if kind == "Reaching" and dist is not None
            else None,
            path_length_m=round(dist, 1) if dist is not None else None,
            radius_m=event.radius_m if event is not None else None,
            continuity=event.continuity if event is not None else None,
            turn_direction=event.turn_direction if event is not None else "",
            outcome=event.outcome if event is not None else "",
            score_scope=event.score_scope if event is not None else "",
            confidence=event.confidence if event is not None else "",
            a=a,
            b=b,
        )
        if event is not None:
            s.directional_score = event.directional_score
            s.pause_s = event.pause_s
            s.counterturn_deg = event.counterturn_deg
            s.setup_s = event.setup_s
            s.rotation_deg = event.rotation_deg
            s.completion_s = event.completion_s
            s.completion_to_stop_s = event.completion_to_stop_s
            s.min_speed_mps = event.min_speed_mps
            s.carve_reversals = event.carve_reversals
            s.heading_range_deg = event.heading_range_deg
            s.mean_course_deg = event.mean_course_deg
            s.crosswind_fraction = event.crosswind_fraction
            s.detection_source = event.detection_source
            s.core_a = event.core_a
            s.core_b = event.core_b
            s.axis_index = event.axis_index
            s.core_start_s = event.core_start_s
            s.core_end_s = event.core_end_s
            s.setup_duration_s = event.setup_duration_s
            s.core_duration_s = event.core_duration_s
            s.exit_observation_s = event.exit_observation_s
            s.core_length_m = event.core_length_m
            s.core_radius_m = event.core_radius_m
            s.core_continuity = event.core_continuity
            s.entry_radius_m = event.entry_radius_m
            s.exit_radius_m = event.exit_radius_m
            s.phase_status = event.phase_status
        seg.append(s)
        a = b
    elapsed = float(session.get("total_elapsed_time", t[-1] - t[0]))
    if elapsed > seg[-1].end_s:
        s = Segment(
            id=len(seg) + 1,
            type="Unknown / gap",
            start_s=seg[-1].end_s,
            end_s=elapsed,
            duration_s=round(elapsed - seg[-1].end_s, 3),
            reach_length_m=None,
            path_length_m=None,
            radius_m=None,
            continuity=None,
            turn_direction="",
            outcome="",
            score_scope="",
            confidence="",
            a=len(t) - 1,
            b=len(t) - 1,
        )
        seg.append(s)
    assert abs(sum(s.duration_s for s in seg) - elapsed) < 0.01
    assert all(abs(x.end_s - y.start_s) < 0.001 for x, y in zip(seg, seg[1:]))
    return seg
