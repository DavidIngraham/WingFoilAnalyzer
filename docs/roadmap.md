# Next milestones

## 1. Make disagreement reviewable

Add aligned heuristic/HSMM timelines to the HTML track report and export rider corrections with session ID, event type, start/end, outcome, uncertainty, and label provenance. Start with recovered gybes, surf/turn disagreements, intentional stops versus crashes, and phase boundaries.

Acceptance: a reviewed session can be exported and reopened without losing corrections; original model predictions remain available for comparison.

## 2. Build independent evidence

Review additional sessions containing tacks, failed maneuvers, stops, varied wind, and sampling gaps. Keep uncertain intervals explicit. Define event-matching and boundary tolerances before scoring.

Acceptance: a versioned label set with session-level development/held-out splits and no same-session leakage. Data sharing requires a separate decision; recordings remain local by default.

## 3. Fit once, evaluate elsewhere

Separate model fitting, serialization, and inference. Freeze normalization, emissions, and duration settings before decoding held-out sessions. Compare against the existing heuristic baseline using event precision/recall, boundary error, outcome agreement, and duration-weighted state agreement.

Acceptance: a reproducible comparison report on unseen sessions. Agreement with weak training labels alone does not qualify as validation.

## 4. Decide whether to promote the HSMM

Define how state paths become maneuver events and how uncertainty and missing data propagate. Add regression cases for sparse data, unknown intervals, counterturns, and absent training classes before changing the default outputs.

Acceptance: evidence of useful improvement over the baseline and a documented fallback for ambiguous cases.

## 5. Explore watch-style inference

Replace full-session fitting with a frozen model, bound decoder memory, and define how labels are committed. Replay prefixes to measure confirmation delay and test long-gap resets. Benchmark on the intended device before selecting a deployment architecture.

Acceptance: measured latency and resource use, plus explicit rules for retrospective labels and final event timestamps.
