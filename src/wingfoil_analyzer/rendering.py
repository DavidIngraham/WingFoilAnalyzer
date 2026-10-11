"""Static map generation and interactive Jinja activity report rendering."""

import pathlib
from collections import Counter

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .types import Crash, FloatArray, FoilOnset, Segment

COLORS = {
    "Reaching": "#2878b5",
    "Surf": "#e14c91",
    "Gybe": "#119c83",
    "Tack": "#9163c5",
    "Slow / recovery": "#e5a238",
    "Stopped / drifting": "#8b94a3",
    "Unknown / gap": "#d1d5db",
}


def clock(seconds: float) -> str:
    """Format elapsed seconds as an HH:MM:SS clock value."""
    value = int(round(seconds))
    return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}"


def _map_points(
    xy: FloatArray,
    start: int,
    end: int,
    lower: FloatArray,
    scale: float,
) -> str:
    """Convert a segment's finite coordinates to overview SVG points."""
    points = xy[start : end + 1]
    return " ".join(
        f"{20 + (point[0] - lower[0]) * scale:.1f},"
        f"{740 - (point[1] - lower[1]) * scale:.1f}"
        for point in points
        if np.isfinite(point).all()
    )


def _mini_points(xy: FloatArray, start: int, end: int) -> str:
    """Normalize one ride's path into a compact 100 by 60 SVG icon."""
    points = xy[start : end + 1]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return ""
    lower = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - lower, 1.0)
    return " ".join(
        f"{8 + (point[0] - lower[0]) / span[0] * 84:.1f},"
        f"{52 - (point[1] - lower[1]) / span[1] * 44:.1f}"
        for point in points
    )


def _segment_view(
    segment: Segment,
    xy: FloatArray,
    lower: FloatArray,
    scale: float,
    elapsed: float,
) -> dict[str, object]:
    """Build the template view model for one timeline segment."""
    duration = max(segment.duration_s, 0.0)
    return {
        "id": segment.id,
        "type": segment.type,
        "color": COLORS.get(segment.type, "#8b94a3"),
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "duration_s": duration,
        "start_clock": clock(segment.start_s),
        "end_clock": clock(segment.end_s),
        "duration_text": f"{duration:.1f}s",
        "points": _map_points(xy, segment.a, segment.b, lower, scale),
        "left_pct": 100 * segment.start_s / elapsed if elapsed else 0,
        "width_pct": 100 * duration / elapsed if elapsed else 0,
        "outcome": segment.outcome,
        "confidence": segment.confidence,
        "direction": segment.turn_direction,
        "continuity": segment.continuity,
        "radius_m": segment.radius_m,
        "path_length_m": segment.path_length_m,
        "core_duration_s": segment.core_duration_s,
        "core_radius_m": segment.core_radius_m,
        "core_continuity": segment.core_continuity,
        "setup_duration_s": segment.setup_duration_s,
        "exit_observation_s": segment.exit_observation_s,
    }


def render(
    out: pathlib.Path,
    t: FloatArray,
    xy: FloatArray,
    seg: list[Segment],
    crash: list[Crash],
    foil: list[FoilOnset],
) -> None:
    """Write the static map and interactive per-ride/timeline HTML report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 10))
    for segment in seg:
        points = xy[segment.a : segment.b + 1] / 1000
        ax.plot(
            points[:, 0],
            points[:, 1],
            c=COLORS.get(segment.type, "#8b94a3"),
            lw=2 if segment.type in ("Gybe", "Tack", "Surf") else 0.85,
            alpha=0.95 if segment.type in ("Gybe", "Surf") else 0.65,
        )
    if crash:
        indices = [event.sample_index for event in crash]
        ax.scatter(xy[indices, 0] / 1000, xy[indices, 1] / 1000, c="#df3434", s=25)
    if foil:
        indices = [event.sample_index for event in foil]
        ax.scatter(xy[indices, 0] / 1000, xy[indices, 1] / 1000, c="#168a5a", s=25)
    ax.set(
        aspect="equal",
        xlabel="East from first valid position (km)",
        ylabel="North from first valid position (km)",
        title="Wingfoil activity overview",
    )
    ax.legend(
        handles=[Line2D([0], [0], color=color, lw=3, label=label) for label, color in COLORS.items()]
        + [Line2D([0], [0], color="#df3434", marker="o", lw=0, label="Probable crash")]
        + [Line2D([0], [0], color="#168a5a", marker="o", lw=0, label="Onto foil")],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "activity_map.png", dpi=170)
    plt.close(fig)

    valid = xy[np.isfinite(xy).all(axis=1)]
    lower = valid.min(axis=0) - 30.0
    upper = valid.max(axis=0) + 30.0
    scale = min(760 / max(upper[0] - lower[0], 1.0), 700 / max(upper[1] - lower[1], 1.0))
    elapsed = float(seg[-1].end_s) if seg else float(t[-1] - t[0])
    segments = [_segment_view(item, xy, lower, scale, elapsed) for item in seg]
    rides = [
        {
            **item,
            "label": f"{item['type']} {item['id']}",
            "mini_points": _mini_points(xy, segment.a, segment.b),
        }
        for item, segment in zip(segments, seg)
        if item["type"] in ("Gybe", "Tack", "Surf")
    ]
    crashes = [
        {
            "id": event.id,
            "time_s": event.time_s,
            "time_text": clock(event.time_s),
            "sample_index": event.sample_index,
            "context": event.context,
            "x": 20 + (xy[event.sample_index, 0] - lower[0]) * scale,
            "y": 740 - (xy[event.sample_index, 1] - lower[1]) * scale,
        }
        for event in crash
    ]
    foil_events = [
        {
            "id": event.id,
            "time_s": event.time_s,
            "time_text": clock(event.time_s),
            "sample_index": event.sample_index,
            "context": event.context,
            "x": 20 + (xy[event.sample_index, 0] - lower[0]) * scale,
            "y": 740 - (xy[event.sample_index, 1] - lower[1]) * scale,
        }
        for event in foil
    ]
    gybes = [item for item in seg if item.type == "Gybe"]
    successful_gybes = sum(item.outcome == "Speed-retaining exit" for item in gybes)
    summary = {
        "elapsed_text": clock(elapsed),
        "segment_count": len(seg),
        "ride_count": len(rides),
        "crash_count": len(crash),
        "foil_count": len(foil),
        "gybe_success_rate": (
            f"{100 * successful_gybes / len(gybes):.1f}%" if gybes else "—"
        ),
        "type_counts": dict(Counter(item.type for item in seg)),
    }
    environment = Environment(
        loader=FileSystemLoader(pathlib.Path(__file__).with_name("templates")),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template("activity_report.html.j2")
    document = template.render(
        summary=summary,
        colors=COLORS,
        segments=segments,
        rides=rides,
        crashes=crashes,
        foil_events=foil_events,
        map_width=800,
        map_height=760,
        elapsed=elapsed,
    )
    (out / "activity_report.html").write_text(document, encoding="utf-8")
