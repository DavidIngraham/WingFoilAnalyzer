"""Type aliases for the records produced by the analysis pipeline."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Record: TypeAlias = dict[str, Any]
RecordList: TypeAlias = list[Record]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.intp]
Session: TypeAlias = Mapping[str, Any]


@dataclass(slots=True)
class Event:
	"""Detected maneuver or sustained surf segment."""

	a: int
	b: int
	type: str
	outcome: str
	turn_direction: str
	continuity: int | None
	score_scope: str
	directional_score: int | None
	pause_s: float | None
	counterturn_deg: float | None
	setup_s: float | None
	radius_m: float | None
	rotation_deg: float | None
	completion_s: float | None
	completion_to_stop_s: float | None
	min_speed_mps: float | None
	confidence: str
	detection_source: str | None = None
	carve_reversals: int | None = None
	heading_range_deg: float | None = None
	mean_course_deg: float | None = None
	crosswind_fraction: float | None = None
	core_a: int | None = None
	core_b: int | None = None
	axis_index: int | None = None
	core_start_s: float | None = None
	core_end_s: float | None = None
	setup_duration_s: float | None = None
	core_duration_s: float | None = None
	exit_observation_s: float | None = None
	core_length_m: float | None = None
	core_radius_m: float | None = None
	core_continuity: int | None = None
	entry_radius_m: float | None = None
	exit_radius_m: float | None = None
	phase_status: str | None = None

	def to_dict(self) -> Record:
		"""Return a JSON- and CSV-friendly representation."""
		return asdict(self)


@dataclass(slots=True)
class Crash:
	"""Discrete probable crash detected from a sustained speed drop."""

	id: str
	type: str
	time_s: float
	sample_index: int
	low_speed_duration_s: float
	speed_before_mps: float
	speed_after_mps: float
	context: str
	confidence: str
	timing_uncertainty_s: float
	onset_lower_s: float | None = None
	confirmed_at_s: float | None = None
	maximum_gap_s: float | None = None
	evidence: str | None = None

	def to_dict(self) -> Record:
		"""Return a JSON- and CSV-friendly representation."""
		return asdict(self)


@dataclass(slots=True)
class FoilOnset:
	"""Discrete transition into sustained powered speed."""

	id: str
	type: str
	time_s: float
	sample_index: int
	speed_mps: float
	speed_before_mps: float
	powered_duration_s: float
	context: str
	confidence: str

	def to_dict(self) -> Record:
		"""Return a JSON- and CSV-friendly representation."""
		return asdict(self)


@dataclass(slots=True)
class Segment:
	"""Contiguous labeled interval in the activity timeline."""

	id: int
	type: str
	start_s: float
	end_s: float
	duration_s: float
	reach_length_m: float | None
	path_length_m: float | None
	radius_m: float | None
	continuity: int | None
	turn_direction: str
	outcome: str
	score_scope: str
	confidence: str
	a: int
	b: int
	directional_score: int | None = None
	pause_s: float | None = None
	counterturn_deg: float | None = None
	setup_s: float | None = None
	rotation_deg: float | None = None
	completion_s: float | None = None
	completion_to_stop_s: float | None = None
	min_speed_mps: float | None = None
	carve_reversals: int | None = None
	heading_range_deg: float | None = None
	mean_course_deg: float | None = None
	crosswind_fraction: float | None = None
	detection_source: str | None = None
	core_a: int | None = None
	core_b: int | None = None
	axis_index: int | None = None
	core_start_s: float | None = None
	core_end_s: float | None = None
	setup_duration_s: float | None = None
	core_duration_s: float | None = None
	exit_observation_s: float | None = None
	core_length_m: float | None = None
	core_radius_m: float | None = None
	core_continuity: int | None = None
	entry_radius_m: float | None = None
	exit_radius_m: float | None = None
	phase_status: str | None = None

	def to_dict(self) -> Record:
		"""Return a JSON- and CSV-friendly representation."""
		return asdict(self)


@dataclass(slots=True)
class SurfMetrics:
	"""Metrics describing sustained crosswind heading oscillation."""

	reversals: int
	heading_range_deg: float
	crosswind_fraction: float
	same_side_fraction: float
	mean_course_deg: float
	ext: list[tuple[int, float]]


@dataclass(slots=True)
class ArcMetrics:
	"""Ground-track and angular metrics for a reliable maneuver interval."""

	length_m: float | None
	radius_m: float | None
	continuity: int | None
