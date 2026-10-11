# Wingfoil Analyzer

## Install and run

This repository is a UV-buildable Python package. Create the environment and run
the analyzer with:

    uv sync
    uv run wingfoil-analyze recording.zip --out results --wind 247.5 --window 3
    uv run wingfoil-analyze recording.zip --out results --crash-detector causal

To build distributable artifacts:

    uv build

The detector tests can be run with:

    uv run python -m unittest discover -s tests -p 'test_*.py'

## Project layout

- `src/wingfoil_analyzer/loading.py`: FIT and ZIP parsing plus coordinate normalization
- `src/wingfoil_analyzer/types.py`: shared record, event, and segment type aliases
- `src/wingfoil_analyzer/analysis.py`: course estimation, detection, segmentation, and output orchestration
- `src/wingfoil_analyzer/phase_refinement.py`: retrospective setup/core/exit maneuver boundaries and phase metrics
- `src/wingfoil_analyzer/causal_crashes.py`: optional prefix-causal riding-to-stop crash detector
- `src/wingfoil_analyzer/hsmm.py`: optional explicit-duration HSMM state challenger
- `src/wingfoil_analyzer/rendering.py`: static map and self-contained HTML report generation
- `src/wingfoil_analyzer/cli.py`: console-script entry point

Analyze an original FIT file or ZIP containing a FIT activity:

    uv run wingfoil-analyze recording.zip --out results --wind 247.5 --window 3
    python -m unittest discover -s tests -p 'test_*.py'

Dependencies: Python 3, numpy, scipy, fitdecode, matplotlib.

This is an exploratory heuristic, not a trained statistical classifier. Locally retained development outputs
are from 24277338530.zip, 2026-09-07, Event Site, Hood River, with user-reported WSW
wind represented as wind FROM 247.5 degrees true. The source activity is not bundled.

## Outputs
- activity_report.html: self-contained full table and clickable color-coded SVG ground-track map
- activity_map.png: static colored map, numbered by table segment IDs
- segments.csv / segments.json: every interval including reaching, gybing, slow/recovery,
  stopped/drifting, and unknown/gap; times are elapsed seconds from recording start
- events.csv / events.json: maneuver metrics and outcome detail
- crashes.csv / crashes.json: probable crash events
- foils.csv / foils.json: discrete onto-foil transitions
- hsmm_states.csv / hsmm_states.json: provisional state labels from `--decoder hsmm`
- summary.json: totals

## Detection
Trailing local linear position fits reconstruct course without resampling. At least two
samples are used, so actual span can exceed the nominal 3-second window. Fits across
gaps over 8 seconds or with speed below 1 m/s are rejected. Recorded FIT speed is retained.
Stable moving approach: at least 3 eligible samples spanning 6 seconds in the previous
12 seconds, >=3 m/s, within 18 degrees of the mean course, away from wind axes.
A 25-degree departure starts a candidate. Up to 6 seconds of preceding setup is included.
Initial direction relative to wind distinguishes gybe from tack. No-tack rider recollection
is not used to force the output. A developed attempt requires >=75 degrees of net signed
rotation from entry, even if it never reaches the other side. Counter-turning and hesitation
remain in the open event. Heading evidence below 2 m/s is excluded.
Opposite-side course, sufficiently away from the wind axis, persisting for 2 seconds marks
geometric completion. This is a coarse course-based proxy, not evidence of board/wing attitude.
Exit observation is 10 seconds from completion. Low speed <1.5 m/s persisting for 5 seconds
is a probable crash. If low speed begins inside the exit window, observation extends to
confirm it. A brief low episode need not be a crash. Other endings: sustained return to
entry heading for 6 seconds, 45-second candidate timeout (sample timing can overshoot), or
end of recording. A candidate remains one event throughout, including its exit observation.

Residual intervals use mean endpoint speed: >=3 m/s reaching; >=1.5 slow/recovery; otherwise
stopped/drifting. Short speed-state islands <6 seconds between matching states are merged.
This residual cleanup and retrospective start labeling make v3 an offline prototype, not a
drop-in streaming watch implementation. Course estimation itself is trailing/causal.
Missing-position samples and gaps >8 seconds outside maneuvers are explicitly unknown.
Maneuvers that span sparse data keep their event identity but receive lower confidence.
The final unrecorded fractional second is unknown, preserving total session elapsed time.

## Metrics
Reach length: sum of successive position distances, excluding gaps >8 seconds. It is a
sampled ground-track length, not water-relative length or the Garmin accumulated distance.
Turn radius: effective radius = sum of reliable position chord lengths / total absolute
course rotation (radians) over setup/turn before the exit observation. This is not a fitted
circle, not a minimum radius, and partial arcs/wobbles can bias it. Needs >=4 reliable samples
and >=45 degrees angular travel; otherwise blank.
Continuity: 70% directional consistency + 30% fraction of observed turn time not stalled.
Directional consistency = 100*absolute net angular change / total angular change.
Angular increments <3 degrees are deadbanded. Pause threshold is <3 degrees/second.
Small increments are treated as pauses; therefore sampling and window length affect scores.
Counterturn_deg reports rotation opposite the initial turn direction. Setup_s reports time
from event start to first 75-degree commitment; it includes early committed rotation and is
not a separately identified pre-turn phase. Partial scores cover incomplete observed turns;
do not rank them against full-turn scores. At least 4 reliable samples and >30 degrees total
rotation are required for a numeric score. Continuity and outcome are intentionally separate.
Direction is clockwise/right or counterclockwise/left in a north-up map (initial direction).
Confidence is a coarse data-quality flag, not a calibrated probability. No high-confidence
labels are assigned. Low includes incomplete turns, >5-second gaps, or fewer than 5 turn samples.

## Phase refinement
Detected gybes and tacks retain their original event boundaries and outcomes, then receive
retrospective setup/core/exit metrics. Core boundaries are selected from sustained signed turn
rate of at least 3 degrees/second, bridging gaps up to 6 seconds unless strong counter-turning
is present. Core radius, continuity, entry radius, and exit radius are engineering metrics, not
physical onset measurements. They use the same reliability guards as the main detector.

## Causal crash mode
The default `offline` crash detector uses the complete recording and preserves the existing
retrospective behavior. `--crash-detector causal` uses only observed prefixes: at least 2 seconds
of riding at >=3.5 m/s arms the detector, a stop below 2.0 m/s must persist for 5 seconds, and
gaps over 30 seconds reset the episode. The emitted event is backdated to the first low-speed
sample and includes its confirmation time. This is suitable for replay/watch-style processing,
but its labels remain probable and are not evidence of a physical fall.

The prototype HSMM remains an experimental challenger rather than a production dependency:
its parameters were weakly fit from one recording, require SciPy, and have no independent
rider-confirmed accuracy validation.

## HSMM challenger
Use `--decoder hsmm` to emit provisional HSMM state labels alongside the normal heuristic
outputs. The model has `Stopped`, `Taxiing`, `Unknown`, directional `Reach` and `Surf`, and
directional `Gybe`/`Tack` `Setup`, `Core`, and `Exit` states. It uses explicit truncated-lognormal
duration survival and a constrained beam decoder. Emission statistics are fit from the current
heuristic labels, so this is a structured comparison tool, not independent validation or a
calibrated probability model. The normal event, crash, foil-onset, and segment outputs remain
heuristic outputs in HSMM mode; the HSMM does not silently replace them.

## Surf classification
New Surf segments identify repeated heading oscillation while average displacement stays
within 40 degrees of crosswind and >=90% of duration stays on one side of the wind.
Windows span up to 36 seconds, require >=24 seconds and >=8 samples, speed >=2 m/s,
no sample gaps >5 seconds, >=3 significant direction reversals (18-degree hysteresis),
and heading range 40-140 degrees. Overlapping windows merge; final fragments must span
>=18 seconds and retain repeated reversals/crosswind motion. This excludes a six-second
setup wobble. Geometry is used throughout the session; no northern geographic fence is
hardcoded. Some additional weaving outside the user-confirmed northern area is therefore
only a surf candidate. No wave-force or wave-height measurements are available.
Confirmed/developed gybe and crash candidates are protected. Unresolved gybe attempts
can be superseded if surf windows cover at least half their duration. Retained maneuvers
split surf windows. Surf is separate from gybe success statistics and has no gybe continuity
score or single turn radius. Its metrics include carve reversals (not number of waves),
heading range, average course, and crosswind displacement fraction.

## Discrete crash detection
A probable crash requires at least two samples >=3.5 m/s spanning >=2 seconds in the
preceding 12 seconds, followed by <1.5 m/s lasting >=5 seconds, with gaps <=8 seconds
within the low-speed run. Each contiguous stop generates at most one event. The timestamp
and red map dot mark the first low-speed sample, not exact impact. The CSV records
sample-interval timing uncertainty, speeds before/after, low-speed duration, and context.
Intentional stops can produce the same signal. Brief breaches with recovery, falls obscured
by recording gaps, and gradual loss of speed may be missed. This is NOT a direct breach
sensor or proof of falling. Discrete crash rows are interleaved with duration segments in
the HTML report; they do not add duration to session coverage. They have separate C IDs.
crashes.csv/json contain the discrete events; segments.csv/json retain contiguous intervals.

## Gybes embedded in surf / variable approaches
A supplemental detector no longer requires steady reaching before a completed turn.
It tracks travel side relative to wind at >=2.5 m/s with |sin(wind-relative course)|>0.6,
then requires 3 seconds of samples on the opposite side. The transition must develop
within 25 seconds, with valid course samples spanning no gaps >8 seconds. The nearest
observed wind-axis course distinguishes gybe from tack. Up to 6 seconds of preceding
setup and 10 seconds of exit observation are retained. Existing developed turns are
not duplicated; overlapping unresolved candidates can be superseded. Surf classification
runs afterward and must split around protected turns: surf -> gybe -> surf/reach.
Recovered events are marked Low confidence pending review, and detection_source records
their origin. Their continuity uses the same angular consistency/pause definitions but
entry boundaries are less certain than stable-approach events. The supplementary check
only recovers completed side changes; failed turns entirely within surf can still be missed.

## Results and validation
32 gybe candidates, zero tacks; 11 surf candidates totaling 546 seconds; 21 probable crashes.
Six gybes were recovered, at geometric-completion elapsed times 12:16, 29:15, 37:07,
1:22:25, 1:41:10, and 1:42:20. Three are at the northern edge. These are candidate
classifications, not ground-truth labels. Gybe outcomes: 23 speed-retaining exits,
five probable crashes during the turn, one after completion, two slow/recovered exits,
and one unresolved slow exit. 267 contiguous duration segments cover 6663.117 seconds.
Synthetic checks cover wobble grouping, crash timing, full coverage, surf distinction,
one crash per stop, surf -> gybe -> surf preservation, and no gybe from same-side carving.
Scores, thresholds, radii and boundaries still require rider/video review.

## Project story and next steps

Read [Reading a wingfoil session as a sequence of maneuvers](docs/reading-a-wingfoil-session.md) and the [milestone plan](docs/roadmap.md).

The FIT/ZIP integration test uses a local activity file and is skipped when that file is absent. Raw recordings, prototype archives, and generated track reports are not published.
