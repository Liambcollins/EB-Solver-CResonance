"""
Tests for the two-segment overhang solver (cant.tip_offset > 0).

Physics under test
------------------
With the tip set back from the free end, the correct model is a two-segment
beam: Bradler tip reaction as interior jump conditions at x_c, and a genuine
free end z''(L) = z'''(L) = 0 at the physical tip.  The former implementation
(terminal BC at x_c + cubic Taylor overhang) violated the free-end conditions
and produced spurious displacement nodes on the overhang ("fold" artifact in
notebook 09).  Verified independently against a Hermite-FEM discretization of
the full beam (July 2026).
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eb_cr_afm import (
    CantileverParams, ContactParams, ExcitationParams, SolverOptions,
    solve_single_frequency,
)

# nb07/nb09 manuscript geometry: tall tip, 15 um overhang
CANT = CantileverParams(L=240e-6, w=40e-6, H=15e-6, k=3.0, m=5e-11, mT=5e-12,
                        phi=0.2, tip_offset=15e-6)
CT = ContactParams(k1=7056.0, k2=0.85 * 7056.0, c1=20e-6, c2=0.0)
EXC_PZ = ExcitationParams(d1=1e-12, d2=0.0, f0=0.0, use_f0=False,
                          bias_coupled_f0=False, delta_V=0.0, V_AC=0.0,
                          eps_r=1.0, use_distributed_load=False)
EXC_ES = ExcitationParams(d1=0.0, d2=0.0, f0=0.0, use_f0=False,
                          bias_coupled_f0=False, delta_V=1.0, V_AC=1.0,
                          eps_r=1.0, use_distributed_load=True)
OPTS = SolverOptions(n_points=600, g=20.0, method='analytical')
F_CR = 529.6e3   # contact resonance for CT (FEM-verified)


def _mask_overhang(x):
    return x > CANT.contact_x + 1e-9


class TestFreeEnd:
    def test_free_end_moment_and_shear_vanish(self):
        res = solve_single_frequency(F_CR, CANT, CT, EXC_PZ, OPTS)
        zpp, zppp = res['zpp'], res['zppp']
        # relative to the (large) tip-reaction values at the contact
        ic = int(np.argmin(np.abs(res['x'] - CANT.contact_x)))
        assert abs(zpp[-1]) < 1e-8 * abs(zpp[ic])
        assert abs(zppp[-1]) < 1e-8 * abs(zppp[ic])

    def test_free_end_with_distributed_load(self):
        res = solve_single_frequency(30e3, CANT, CT, EXC_ES, OPTS)
        ic = int(np.argmin(np.abs(res['x'] - CANT.contact_x)))
        assert abs(res['zpp'][-1]) < 1e-6 * abs(res['zpp'][ic])
        assert abs(res['zppp'][-1]) < 1e-6 * abs(res['zppp'][ic])


class TestContinuityAndJump:
    def test_z_and_zp_continuous_at_contact(self):
        res = solve_single_frequency(F_CR, CANT, CT, EXC_PZ, OPTS)
        x, z, zp = res['x'], res['z'], res['zp']
        ic = int(np.argmin(np.abs(x - CANT.contact_x)))
        # steps across the junction must be comparable to neighbouring steps
        step_before = abs(z[ic] - z[ic - 1])
        step_across = abs(z[ic + 1] - z[ic])
        dx_before = x[ic] - x[ic - 1]
        dx_across = x[ic + 1] - x[ic]
        assert step_across / dx_across < 10 * max(step_before / dx_before, 1e-30)
        sp_before = abs(zp[ic] - zp[ic - 1]) / dx_before
        sp_across = abs(zp[ic + 1] - zp[ic]) / dx_across
        assert sp_across < 10 * max(sp_before, 1e-30)

    def test_moment_jump_balances_tip_reaction(self):
        from eb_cr_afm.solver import (compute_alpha4, compute_contact_stiffnesses,
                                      compute_auxiliary, compute_tip_forces)
        res = solve_single_frequency(F_CR, CANT, CT, EXC_PZ, OPTS)
        x = res['x']
        ic = int(np.argmin(np.abs(x - CANT.contact_x)))
        omega = 2 * np.pi * F_CR
        alpha4 = compute_alpha4(omega, CANT, OPTS)
        k1c, k2c = compute_contact_stiffnesses(omega, CT)
        T, X, U = compute_auxiliary(alpha4, k1c, k2c, CANT, CT)
        F_vert, F_lon = compute_tip_forces(omega, k1c, k2c, EXC_PZ)
        EI, H = CANT.EI, CANT.H
        th = CANT.phi + CT.psi
        RHS_mom = (H / EI) * np.cos(th) * F_lon - (H / EI) * np.sin(th) * F_vert
        # overhang-side curvature at x_c, extrapolated first-order from ic+1
        zpp_plus = res['zpp'][ic + 1] - res['zppp'][ic + 1] * (x[ic + 1] - x[ic])
        jump = res['zpp'][ic] - zpp_plus
        expected = RHS_mom - (T * res['zp'][ic] + X * res['z'][ic])
        assert abs(jump - expected) < 5e-3 * max(abs(expected), abs(res['zpp'][ic]))


class TestBradlerLimit:
    def test_tip_offset_to_zero_recovers_terminal_solution(self):
        """L2 -> 0 must reduce exactly to the Bradler terminal BCs."""
        cant0 = CantileverParams(L=240e-6, w=40e-6, H=15e-6, k=3.0, m=5e-11,
                                 mT=5e-12, phi=0.2, tip_offset=0.0)
        cant_eps = CantileverParams(L=240e-6, w=40e-6, H=15e-6, k=3.0, m=5e-11,
                                    mT=5e-12, phi=0.2, tip_offset=1e-9)
        r0 = solve_single_frequency(350e3, cant0, CT, EXC_PZ, OPTS)
        re = solve_single_frequency(350e3, cant_eps, CT, EXC_PZ, OPTS)
        for xq in (60e-6, 120e-6, 200e-6, 239e-6):
            z0 = np.interp(xq, r0['x'], np.real(r0['z'])) \
                 + 1j * np.interp(xq, r0['x'], np.imag(r0['z']))
            ze = np.interp(xq, re['x'], np.real(re['z'])) \
                 + 1j * np.interp(xq, re['x'], np.imag(re['z']))
            assert abs(z0 - ze) < 1e-4 * abs(z0)


class TestBVPCrossCheck:
    def test_analytical_matches_bvp_two_segment(self):
        opts_b = SolverOptions(n_points=300, g=20.0, method='bvp')
        opts_a = SolverOptions(n_points=300, g=20.0, method='analytical')
        ra = solve_single_frequency(350e3, CANT, CT, EXC_PZ, opts_a)
        rb = solve_single_frequency(350e3, CANT, CT, EXC_PZ, opts_b)
        ref = np.max(np.abs(ra['z']))
        for xq in (60e-6, 150e-6, 220e-6, 232e-6, 239.9e-6):
            za = np.interp(xq, ra['x'], np.real(ra['z'])) \
                 + 1j * np.interp(xq, ra['x'], np.imag(ra['z']))
            zb = np.interp(xq, rb['x'], np.real(rb['z'])) \
                 + 1j * np.interp(xq, rb['x'], np.imag(rb['z']))
            assert abs(za - zb) < 1e-2 * ref

    def test_analytical_matches_bvp_with_distributed_load(self):
        opts_b = SolverOptions(n_points=300, g=20.0, method='bvp')
        opts_a = SolverOptions(n_points=300, g=20.0, method='analytical')
        ra = solve_single_frequency(30e3, CANT, CT, EXC_ES, opts_a)
        rb = solve_single_frequency(30e3, CANT, CT, EXC_ES, opts_b)
        ref = np.max(np.abs(ra['z']))
        for xq in (60e-6, 150e-6, 220e-6, 232e-6, 239.9e-6):
            za = np.interp(xq, ra['x'], np.real(ra['z'])) \
                 + 1j * np.interp(xq, ra['x'], np.imag(ra['z']))
            zb = np.interp(xq, rb['x'], np.real(rb['z'])) \
                 + 1j * np.interp(xq, rb['x'], np.imag(rb['z']))
            assert abs(za - zb) < 1e-2 * ref


class TestNoSpuriousFold:
    def test_single_smooth_node_in_fold_region(self):
        """k1 = 7056, k2 = 0.85 k1 sat mid-'fold' in the old model: it showed
        two overhang nodes (x/L ~ 0.965, 0.985).  The correct model has exactly
        one deep null near x/L = 0.9554 (FEM-verified) and nothing beyond 0.97.
        """
        from scipy.signal import argrelmin
        res = solve_single_frequency(F_CR, CANT, CT, EXC_PZ, OPTS)
        x, z = res['x'], res['z']
        m = x >= CANT.L - 70e-6
        xs = np.linspace(x[m][0], x[-1], 30001)
        A = np.abs(np.interp(xs, x[m], np.real(z[m]))
                   + 1j * np.interp(xs, x[m], np.imag(z[m])))
        idx = argrelmin(A)[0]
        deep = [xs[i] / CANT.L for i in idx if A[i] / A.max() < 0.03]
        assert len(deep) == 1
        assert abs(deep[0] - 0.9554) < 0.003
        assert not any(d > 0.97 for d in deep)


class TestOverhangWidthRatio:
    """The overhang electrostatic width controls where the D-ESBS sits.

    Full-width overhang (ratio 1.0) over-loads the overhang and pulls the ES
    displacement null toward the contact; a realistic narrow cone/tip region
    (ratio << 1) moves it out toward the free end, reproducing the ordering of
    realistic finite-element models. The ratio must not touch the piezo
    (mechanics-only) response.
    """
    GEOM = dict(L=225e-6, w=28e-6, H=12e-6, k=2.8, m=5.2e-11, mT=0.0,
                phi=np.radians(11.0), tip_offset=15e-6)
    CT = ContactParams(k1=1e6, k2=1e6, c1=0.0, c2=0.0)

    def _es_null_from_end(self, ratio):
        cant = CantileverParams(overhang_width_ratio=ratio, **self.GEOM)
        r = solve_single_frequency(10e3, cant, self.CT, EXC_ES, OPTS)
        x = r['x']; m = x >= cant.L - 75e-6
        xs = np.linspace(x[m][0], x[-1], 8000)
        A = np.abs(np.interp(xs, x[m], np.real(r['z'][m]))
                   + 1j * np.interp(xs, x[m], np.imag(r['z'][m])))
        return (cant.L - xs[int(np.argmin(A))]) * 1e6   # um from free end

    def test_default_is_full_width(self):
        assert CantileverParams(**self.GEOM).overhang_width_ratio == 1.0

    def test_ratio_does_not_change_piezo_response(self):
        c1 = CantileverParams(overhang_width_ratio=1.0, **self.GEOM)
        c2 = CantileverParams(overhang_width_ratio=0.05, **self.GEOM)
        r1 = solve_single_frequency(3e5, c1, self.CT, EXC_PZ, OPTS)
        r2 = solve_single_frequency(3e5, c2, self.CT, EXC_PZ, OPTS)
        assert np.allclose(r1['z'], r2['z'], rtol=1e-9, atol=0)

    def test_narrow_overhang_moves_esbs_toward_tip(self):
        # D-NS (piezo) for this geometry sits ~11.8 um from the end.
        full = self._es_null_from_end(1.0)     # contact side (~14.7)
        narrow = self._es_null_from_end(0.05)  # tip side (~3.5)
        assert full > 13.0            # contact side of the D-NS
        assert narrow < 6.0           # tip side of the D-NS
        assert narrow < full - 5.0    # monotone, large shift


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
