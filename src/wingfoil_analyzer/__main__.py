"""Command-line orchestration for the wingfoil analyzer."""

import argparse
import collections
import csv
import json
import pathlib

from .analysis import (
    add_surf,
    courses,
    crashes,
    events,
    foil_onsets,
    recover_transitions,
    segment,
)
from .causal_crashes import detect_crashes
from .hsmm import (
    decode as hsmm_decode,
)
from .hsmm import (
    features as hsmm_features,
)
from .hsmm import (
    fit_model,
    state_records,
    states_and_edges,
    weak_labels,
)
from .loading import load
from .phase_refinement import refine_phases
from .rendering import COLORS, render


def main() -> None:
    """Parse command-line arguments and run the complete analysis pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="results")
    parser.add_argument("--wind", type=float, default=247.5)
    parser.add_argument("--window", type=float, default=3)
    parser.add_argument(
        "--decoder",
        choices=("heuristic", "hsmm"),
        default="heuristic",
        help="Choose the production heuristic or experimental HSMM state decoder.",
    )
    parser.add_argument(
        "--crash-detector",
        choices=("offline", "causal"),
        default="offline",
        help="Choose retrospective or prefix-causal crash detection.",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _rows, t, xy, v, session = load(args.input)
    c = courses(t, xy, args.window)
    ev = add_surf(
        t,
        c,
        v,
        args.wind,
        xy,
        recover_transitions(t, c, v, args.wind, xy, events(t, c, v, args.wind, xy)),
    )
    ev = refine_phases(t, c, v, xy, args.wind, ev)
    seg = segment(t, xy, v, ev, session)
    hsmm_records: list[dict[str, object]] = []
    hsmm_counts: dict[str, int] = {}
    if args.decoder == "hsmm":
        hsmm_states, _edges = states_and_edges()
        labels = weak_labels(t, seg, ev)
        model = fit_model(hsmm_features(t, c, v, args.wind), labels)
        decoded = hsmm_decode(t, c, v, args.wind, model)
        hsmm_records = state_records(t, decoded, hsmm_states)
        hsmm_counts = {
            str(state): count
            for state, count in collections.Counter(
                record["state"] for record in hsmm_records
            ).items()
        }
    crash = (
        crashes(t, v, xy, ev)
        if args.crash_detector == "offline"
        else detect_crashes(t, v)
    )
    foil = foil_onsets(t, v, xy, ev)
    for name, data in [
        ("segments", seg),
        ("events", ev),
        ("crashes", crash),
        ("foils", foil),
    ]:
        records = [row.to_dict() for row in data]
        (out / (name + ".json")).write_text(json.dumps(records, indent=2))
        if records:
            with (out / (name + ".csv")).open("w") as file:
                writer = csv.DictWriter(
                    file, fieldnames=list(dict.fromkeys(k for row in records for k in row))
                )
                writer.writeheader()
                writer.writerows(records)
    if hsmm_records:
        (out / "hsmm_states.json").write_text(json.dumps(hsmm_records, indent=2))
        with (out / "hsmm_states.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(hsmm_records[0]))
            writer.writeheader()
            writer.writerows(hsmm_records)
    render(out, t, xy, seg, crash, foil)
    gybes = [event for event in ev if event.type == "Gybe"]
    successful_gybes = sum(
        event.outcome == "Speed-retaining exit" for event in gybes
    )
    summary = {
        "probable_crashes": len(crash),
        "crash_detector": args.crash_detector,
        "decoder": args.decoder,
        "foil_onsets": len(foil),
        "segments": len(seg),
        "event_count": len(ev),
        "gybe_success_rate_pct": round(100 * successful_gybes / len(gybes), 1)
        if gybes
        else None,
        "hsmm_state_counts": hsmm_counts if args.decoder == "hsmm" else None,
        "types": dict(collections.Counter(e.type for e in ev)),
        "outcomes": dict(collections.Counter(e.outcome for e in ev)),
        "duration_s": sum(s.duration_s for s in seg),
        "activity_duration_s": session.get("total_elapsed_time"),
        "by_activity": {
            k: {
                "segments": sum(s.type == k for s in seg),
                "seconds": round(sum(s.duration_s for s in seg if s.type == k), 1),
            }
            for k in COLORS
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()