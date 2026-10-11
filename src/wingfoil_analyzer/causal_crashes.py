"""Causal riding-to-stop crash detection.

This detector confirms events only from samples already observed. It is useful
for replay or watch-style processing; the existing offline detector remains the
default for retrospective analysis.
"""

import math
from dataclasses import dataclass

from .types import Crash, FloatArray


@dataclass(frozen=True, slots=True)
class CrashConfig:
    """Thresholds governing causal riding and stop confirmation."""

    riding_mps: float = 3.5
    riding_confirm_s: float = 2.0
    stop_enter_mps: float = 2.0
    stop_cancel_mps: float = 2.5
    stop_confirm_s: float = 5.0
    max_gap_s: float = 30.0
    riding_expiry_s: float = 60.0


@dataclass(slots=True)
class _PendingStop:
    """Internal state for a stop awaiting confirmation."""

    time_s: float
    sample_index: int
    speed_after_mps: float
    speed_before_mps: float
    onset_lower_s: float
    maximum_gap_s: float
    last_high_s: float


class CrashDetector:
    """Stateful causal detector that emits at most one event per stop."""

    def __init__(self, config: CrashConfig | None = None) -> None:
        self.config = config or CrashConfig()
        self.previous_s: float | None = None
        self.armed = False
        self.high_start_s: float | None = None
        self.last_high_s: float | None = None
        self.high_speed_mps: float | None = None
        self.pending: _PendingStop | None = None
        self.count = 0

    def update(
        self, time_s: float, speed_mps: float, sample_index: int
    ) -> Crash | None:
        """Consume one sample and return a crash only when confirmation completes."""
        config = self.config
        now = float(time_s)
        speed = float(speed_mps)
        if self.previous_s is not None and now <= self.previous_s:
            raise ValueError("Finite sample times must increase")
        if not math.isfinite(speed):
            self.previous_s = now
            return None
        gap = 0.0 if self.previous_s is None else now - self.previous_s
        if gap > config.max_gap_s:
            self._reset_episode()
        if (
            self.armed
            and self.last_high_s is not None
            and now - self.last_high_s > config.riding_expiry_s
        ):
            self._reset_episode()
        if speed >= config.riding_mps:
            if self.high_start_s is None:
                self.high_start_s = now
            self.last_high_s = now
            self.high_speed_mps = speed
            if now - self.high_start_s >= config.riding_confirm_s:
                self.armed = True
        else:
            self.high_start_s = None
        if speed >= config.stop_cancel_mps:
            self.pending = None
        if self.armed and speed < config.stop_enter_mps and self.pending is None:
            self.pending = _PendingStop(
                time_s=now,
                sample_index=int(sample_index),
                speed_after_mps=speed,
                speed_before_mps=self.high_speed_mps or speed,
                onset_lower_s=self.previous_s if self.previous_s is not None else now,
                maximum_gap_s=gap,
                last_high_s=self.last_high_s or now,
            )
        pending = self.pending
        if pending is None:
            self.previous_s = now
            return None
        pending.maximum_gap_s = max(pending.maximum_gap_s, gap)
        if speed >= config.stop_enter_mps:
            self.previous_s = now
            return None
        if now - pending.time_s < config.stop_confirm_s:
            self.previous_s = now
            return None
        self.count += 1
        sparse = (
            pending.maximum_gap_s > 5
            or pending.time_s - pending.last_high_s > 12
        )
        event = Crash(
            id=f"C{self.count}",
            type="Probable crash",
            time_s=pending.time_s,
            sample_index=pending.sample_index,
            low_speed_duration_s=round(now - pending.time_s, 1),
            speed_before_mps=round(pending.speed_before_mps, 2),
            speed_after_mps=round(pending.speed_after_mps, 2),
            context="Outside detected maneuver",
            confidence="Low" if sparse else "Moderate",
            timing_uncertainty_s=round(
                pending.time_s - pending.onset_lower_s, 1
            ),
            onset_lower_s=pending.onset_lower_s,
            confirmed_at_s=now,
            maximum_gap_s=pending.maximum_gap_s,
            evidence="Established riding followed by sustained stop",
        )
        self._reset_episode()
        self.previous_s = now
        return event

    def _reset_episode(self) -> None:
        """Disarm the detector until a fresh riding episode is established."""
        self.armed = False
        self.high_start_s = None
        self.last_high_s = None
        self.high_speed_mps = None
        self.pending = None


def detect_crashes(
    t: FloatArray, v: FloatArray, config: CrashConfig | None = None
) -> list[Crash]:
    """Run the causal detector over a complete sample sequence."""
    detector = CrashDetector(config)
    output: list[Crash] = []
    for index, (time_s, speed_mps) in enumerate(zip(t, v)):
        event = detector.update(float(time_s), float(speed_mps), index)
        if event is not None:
            output.append(event)
    return output