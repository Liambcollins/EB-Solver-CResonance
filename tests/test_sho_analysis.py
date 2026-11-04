"""Regression tests for the SHO peak fit in sho_analysis()."""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eb_cr_afm import (ContactParams, ExcitationParams, default_params_multi75,
                       frequency_sweep, sho_analysis)


def test_free_cantilever_q_matches_air_damping():
    """Free lever: Q must equal omega/g (the fit must not stall at its guess)."""
    cant, _, _, opts = default_params_multi75()
    free = ContactParams(k1=0.0, k2=0.0, c1=0.0, c2=0.0)
    drive = ExcitationParams(d1=0.0, f0=1e-9, use_f0=True,
                             V_AC=0.0, delta_V=0.0, use_distributed_load=False)
    o = dataclasses.replace(opts, f_start=70e3, f_stop=80e3, f_step=10.0)
    for quantity in ("slope", "displacement"):
        modes = sho_analysis(frequency_sweep(cant, free, drive, o), quantity=quantity)
        assert len(modes) == 1
        m = modes[0]
        q_expected = 2 * np.pi * m["f_res"] / opts.g
        assert m["converged"]
        assert abs(m["f_res"] - 75e3) < 100.0
        assert abs(m["Q"] - q_expected) / q_expected < 0.02


def test_antiresonance_is_not_reported_as_a_mode():
    """The slope anti-resonance near 205 kHz (Multi75-G, k1 = 100 N/m) is not a peak."""
    cant, contact, exc, opts = default_params_multi75()
    sweep = frequency_sweep(cant, contact, exc, opts)
    for quantity in ("slope", "displacement"):
        freqs = [m["f_res"] for m in sho_analysis(sweep, quantity=quantity)]
        assert not any(195e3 < f < 215e3 for f in freqs), freqs
        assert any(abs(f - 288e3) < 1e3 for f in freqs), freqs
