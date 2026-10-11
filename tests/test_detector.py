import unittest

import numpy as np

from wingfoil_analyzer import detect_crashes, events, foil_onsets, segment
from wingfoil_analyzer.hsmm import (
    decode as hsmm_decode,
)
from wingfoil_analyzer.hsmm import (
    features as hsmm_features,
)
from wingfoil_analyzer.hsmm import (
    fit_model,
    states_and_edges,
    survival,
)
from wingfoil_analyzer.phase_refinement import refine_phases
from wingfoil_analyzer.types import FloatArray


def fixture(
    wobble: bool = False, crash: str | None = None
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build a synthetic activity with an optional wobble or crash pattern."""
    t = np.arange(90, dtype=float)
    heading = np.full(90, 120.0)
    path = (
        [120, 135, 150, 135, 145, 160, 180, 200, 220, 240]
        if wobble
        else list(np.linspace(120, 240, 11))
    )
    heading[20 : 20 + len(path)] = path
    heading[20 + len(path) :] = 240
    v = np.full(90, 6.0)
    if crash == "during":
        heading[27:] = 200
        v[28:] = 0.4
    if crash == "after":
        v[36:] = 0.4
    c = np.deg2rad(heading)
    xy = np.cumsum(np.column_stack((np.sin(c) * v, np.cos(c) * v)), axis=0)
    return t, c, v, xy


class DetectorTest(unittest.TestCase):
    def test_wobble_is_single_and_penalized(self) -> None:
        a = fixture()
        b = fixture(True)
        clean = events(a[0], a[1], a[2], 0, a[3])
        wob = events(b[0], b[1], b[2], 0, b[3])
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(wob), 1)
        self.assertLess(wob[0].continuity, clean[0].continuity)

    def test_crash_timing(self) -> None:
        for when, outcome in [
            ("during", "Probable crash during turn"),
            ("after", "Probable crash after completion"),
        ]:
            t, c, v, xy = fixture(crash=when)
            e = events(t, c, v, 0, xy)
            self.assertEqual(len(e), 1)
            self.assertEqual(e[0].outcome, outcome)

    def test_full_coverage(self) -> None:
        t, c, v, xy = fixture()
        e = events(t, c, v, 0, xy)
        s = segment(t, xy, v, e, {"total_elapsed_time": 89.5})
        self.assertAlmostEqual(sum(x.duration_s for x in s), 89.5)
        self.assertTrue(all(x.end_s == y.start_s for x, y in zip(s, s[1:])))

    def test_foil_onset_requires_persistent_powered_speed(self) -> None:
        t = np.arange(8, dtype=float)
        v = np.array([1, 1, 4, 4, 4, 1, 4, 4], dtype=float)
        xy = np.zeros((len(t), 2), dtype=float)
        result = foil_onsets(t, v, xy, [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].time_s, 2)
        self.assertEqual(result[0].speed_before_mps, 1)
        self.assertEqual(result[0].powered_duration_s, 2)

    def test_causal_crash_requires_riding_and_confirmation(self) -> None:
        t = np.arange(20, dtype=float)
        v = np.r_[np.full(5, 6.0), np.full(10, 0.4), np.full(5, 6.0)]
        result = detect_crashes(t, v)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].time_s, 5)
        self.assertEqual(result[0].confirmed_at_s, 10)

    def test_causal_crash_rearms_after_recovery(self) -> None:
        t = np.arange(40, dtype=float)
        v = np.r_[
            np.full(5, 6.0),
            np.full(6, 0.4),
            np.full(8, 6.0),
            np.full(8, 0.4),
            np.full(13, 6.0),
        ]
        self.assertEqual(len(detect_crashes(t, v)), 2)

    def test_phase_refinement_adds_core_boundaries(self) -> None:
        t, c, v, xy = fixture()
        result = refine_phases(t, c, v, xy, 0, events(t, c, v, 0, xy))
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].core_a)
        self.assertIsNotNone(result[0].core_b)
        self.assertIsNotNone(result[0].core_duration_s)

    def test_hsmm_topology_and_decode_are_well_formed(self) -> None:
        states, edges = states_and_edges()
        names = {state.name for state in states}
        self.assertEqual(len(states), 19)
        self.assertIn("Gybe_+1_Core", names)
        self.assertIn("Gybe_+1_Exit", names)
        core = next(state for state in states if state.name == "Gybe_+1_Core")
        self.assertEqual(survival(0, core), 1)
        core_index = next(i for i, state in enumerate(states) if state is core)
        exit_index = next(i for i, state in enumerate(states) if state.name == "Gybe_+1_Exit")
        self.assertIn(exit_index, edges[core_index])

        t = np.arange(35, dtype=float)
        c = np.deg2rad(np.linspace(100, 250, len(t)))
        v = np.full(len(t), 6.0)
        values = hsmm_features(t, c, v, 0)
        labels = np.tile(np.arange(7), 5)
        model = fit_model(values, labels)
        decoded = hsmm_decode(t, c, v, 0, model, beam=16)
        self.assertEqual(decoded.shape, (len(t),))


class SurfAndCrashTest(unittest.TestCase):
    def test_surf_crosswind_vs_short_wobble(self) -> None:
        from wingfoil_analyzer import add_surf

        t = np.arange(100, dtype=float)
        v = np.full(100, 6.0)
        c = np.deg2rad(90 + 30 * np.sin(2 * np.pi * t / 12))
        xy = np.cumsum(np.column_stack((np.sin(c) * v, np.cos(c) * v)), axis=0)
        surf = add_surf(t, c, v, 0, xy, [])
        self.assertTrue(any(e.type == "Surf" for e in surf))
        downwind = c + np.pi / 2
        xy2 = np.cumsum(
            np.column_stack((np.sin(downwind) * v, np.cos(downwind) * v)), axis=0
        )
        self.assertEqual(add_surf(t, downwind, v, 0, xy2, []), [])
        t, c, v, xy = fixture(True)
        ev = events(t, c, v, 0, xy)
        self.assertEqual([e.type for e in add_surf(t, c, v, 0, xy, ev)], ["Gybe"])

    def test_discrete_crash_one_per_stop(self) -> None:
        from wingfoil_analyzer import crashes

        t, c, v, xy = fixture(crash="after")
        ev = events(t, c, v, 0, xy)
        cr = crashes(t, v, xy, ev)
        self.assertEqual(len(cr), 1)
        self.assertEqual(cr[0].context, "After turn completion")
        v[:] = 6
        v[40:43] = 0.4
        self.assertEqual(crashes(t, v, xy, []), [])


class SurfTransitionTest(unittest.TestCase):
    def test_surf_gybe_surf(self) -> None:
        from wingfoil_analyzer import add_surf, recover_transitions

        t = np.arange(180, dtype=float)
        v = np.full(180, 6.0)
        h = 90 + 30 * np.sin(2 * np.pi * t / 12)
        h[60:73] = np.linspace(90, 270, 13)
        h[73:] = 270 + 30 * np.sin(2 * np.pi * (t[73:] - 73) / 12)
        c = np.deg2rad(h)
        xy = np.cumsum(np.column_stack((np.sin(c) * v, np.cos(c) * v)), axis=0)
        e = recover_transitions(t, c, v, 0, xy, [])
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0].type, "Gybe")
        result = add_surf(t, c, v, 0, xy, e)
        self.assertEqual(sum(q.type == "Gybe" for q in result), 1)
        self.assertTrue(
            any(q.type == "Surf" and q.b <= e[0].a for q in result)
        )
        self.assertTrue(
            any(q.type == "Surf" and q.a >= e[0].b for q in result)
        )

    def test_carving_does_not_create_side_change(self) -> None:
        from wingfoil_analyzer import recover_transitions

        t = np.arange(100, dtype=float)
        v = np.full(100, 6.0)
        c = np.deg2rad(90 + 35 * np.sin(2 * np.pi * t / 12))
        xy = np.cumsum(np.column_stack((np.sin(c) * v, np.cos(c) * v)), axis=0)
        self.assertEqual(recover_transitions(t, c, v, 0, xy, []), [])


if __name__ == "__main__":
    unittest.main()