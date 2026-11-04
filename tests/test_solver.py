"""
Unit tests for eb_cr_afm.  Run with: pytest tests/ -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eb_cr_afm import (
    CantileverParams, ContactParams, ExcitationParams, SolverOptions,
    solve_single_frequency, frequency_sweep, analytical_conditioning_sweep,
    sho_analysis, default_params,
    compute_alpha4, compute_contact_stiffnesses, compute_tip_forces,
)

@pytest.fixture
def base_params():
    return default_params()

@pytest.fixture
def simple_cant():
    return CantileverParams(L=200e-6, w=40e-6, H=10e-6,
                            k=1.0, m=40e-9, mT=0.0, phi=0.0)

class TestParams:
    def test_EI(self, simple_cant):
        assert abs(simple_cant.EI - simple_cant.k * simple_cant.L**3/3) < 1e-30

    def test_contact_position_from_tip_offset(self):
        cant = CantileverParams(L=240e-6, tip_offset=15e-6)
        assert cant.contact_x == pytest.approx(225e-6)
        assert cant.L1 == pytest.approx(225e-6)
        assert cant.L2 == pytest.approx(15e-6)

    def test_table_I(self, base_params):
        cant, contact, exc, opts = base_params
        assert cant.L == pytest.approx(240e-6)
        assert cant.k == pytest.approx(3.0)
        assert contact.k1 == pytest.approx(1000.0)
        assert contact.k2/contact.k1 == pytest.approx(0.85, rel=1e-3)
        assert exc.d1 == pytest.approx(1e-12)

    def test_freq_monotonic(self, base_params):
        _, _, _, opts = base_params
        assert np.all(np.diff(opts.frequencies) > 0)

class TestPhysicsHelpers:
    def test_alpha4_zero(self, base_params):
        cant, _, _, opts = base_params
        assert abs(compute_alpha4(0.0, cant, opts)) < 1e-30

    def test_alpha4_imag_sign(self, base_params):
        cant, _, _, opts = base_params
        assert np.imag(compute_alpha4(2*np.pi*350e3, cant, opts)) < 0

    def test_k1c_complex(self, base_params):
        _, contact, _, _ = base_params
        k1c, _ = compute_contact_stiffnesses(2*np.pi*350e3, contact)
        assert np.imag(k1c) > 0

    def test_k1c_real_nodamp(self):
        c = ContactParams(k1=1000.0, k2=850.0, c1=0.0, c2=0.0)
        k1c, _ = compute_contact_stiffnesses(2*np.pi*350e3, c)
        assert abs(np.imag(k1c)) < 1e-10

    def test_bias_coupled_f0_scales_with_bias_squared(self, base_params):
        _, contact, _, _ = base_params
        omega = 2*np.pi*350e3
        k1c, k2c = compute_contact_stiffnesses(omega, contact)

        exc_low = ExcitationParams(
            d1=0.0, d2=0.0,
            f0=0.0,
            delta_V=1.0, V_AC=1.0,
            use_distributed_load=False,
            use_f0=False,
            bias_coupled_f0=True,
            capacitance_gradient=50e-9,
        )
        exc_high = ExcitationParams(
            d1=0.0, d2=0.0,
            f0=0.0,
            delta_V=2.0, V_AC=2.0,
            use_distributed_load=False,
            use_f0=False,
            bias_coupled_f0=True,
            capacitance_gradient=50e-9,
        )

        F_low, _ = compute_tip_forces(omega, k1c, k2c, exc_low)
        F_high, _ = compute_tip_forces(omega, k1c, k2c, exc_high)

        assert F_high == pytest.approx(4.0 * F_low)

    def test_bias_coupled_f0_adds_to_direct_tip_force(self, base_params):
        _, contact, _, _ = base_params
        omega = 2*np.pi*350e3
        k1c, k2c = compute_contact_stiffnesses(omega, contact)

        exc = ExcitationParams(
            d1=0.0, d2=0.0,
            f0=10e-12,
            delta_V=1.0, V_AC=1.0,
            use_distributed_load=False,
            use_f0=True,
            bias_coupled_f0=True,
            capacitance_gradient=50e-9,
        )
        F_vert, _ = compute_tip_forces(omega, k1c, k2c, exc)
        expected = 10e-12 + 0.5 * 50e-9 * (1.0 + 1.0)**2
        assert F_vert == pytest.approx(expected)

class TestSingleFreq:
    def test_bc_zero(self, simple_cant):
        free = ContactParams(k1=0.01, k2=0.0085, c1=1e-12, c2=0.85e-12)
        opts = SolverOptions(n_points=100, f_start=80e3, f_stop=120e3, f_step=2e3, g=50.0)
        exc = ExcitationParams(d1=1e-12, d2=0.0, f0=1e-11,
                               delta_V=0.0, V_AC=0.0, eps_r=1.0,
                               use_distributed_load=False, use_f0=True)
        res = solve_single_frequency(100e3, simple_cant, free, exc, opts)
        assert abs(res["z"][0]) < 1e-20
        assert abs(res["zp"][0]) < 1e-20

    def test_amplitude_positive(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=150, f_start=340e3, f_stop=360e3, f_step=2e3, g=20.0)
        assert solve_single_frequency(350e3, cant, contact, exc, opts)["A"] > 0

    def test_phase_range(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=150, f_start=340e3, f_stop=360e3, f_step=2e3, g=20.0)
        ph = solve_single_frequency(350e3, cant, contact, exc, opts)["phase"]
        assert -np.pi <= ph <= np.pi

    def test_tip_offset_changes_dynamic_endpoint(self, base_params):
        cant, contact, exc, _ = base_params
        tip_offset = 20e-6
        cant_offset = CantileverParams(
            L=cant.L, w=cant.w, H=cant.H, k=cant.k,
            m=cant.m, mT=cant.mT, phi=cant.phi,
            tip_offset=tip_offset,
        )
        opts = SolverOptions(n_points=120, f_start=340e3, f_stop=360e3,
                             f_step=2e3, g=20.0, method="analytical")
        res = solve_single_frequency(350e3, cant_offset, contact, exc, opts)
        # The two-segment solve extends x to cant.L (free overhang past contact).
        assert res["x"][-1] == pytest.approx(cant.L)
        # The contact point (cant.L - tip_offset) must still be within the array.
        assert np.any(np.isclose(res["x"], cant.L - tip_offset, atol=1e-7))

class TestStiffContact:
    def test_resonance_increases_with_k1(self, simple_cant):
        opts = SolverOptions(n_points=80, f_start=50e3, f_stop=500e3, f_step=20e3,
                             g=50.0, method="bvp")
        exc = ExcitationParams(d1=1e-12, d2=0.0, f0=1e-10,
                               delta_V=0.0, V_AC=0.0, eps_r=1.0,
                               use_distributed_load=False, use_f0=True)
        cs = ContactParams(k1=10.0, k2=8.5, c1=1e-9, c2=0.85e-9)
        ch = ContactParams(k1=5000.0, k2=4250.0, c1=1e-9, c2=0.85e-9)
        fs = sho_analysis(frequency_sweep(simple_cant, cs, exc, opts))[0]["f_res"]
        fh = sho_analysis(frequency_sweep(simple_cant, ch, exc, opts))[0]["f_res"]
        assert fh > fs

class TestGridConvergence:
    def test_converges(self, base_params):
        cant, contact, exc, _ = base_params
        rc = solve_single_frequency(350e3, cant, contact, exc,
            SolverOptions(n_points=60, f_start=340e3, f_stop=360e3, f_step=5e3, g=20.0))
        rf = solve_single_frequency(350e3, cant, contact, exc,
            SolverOptions(n_points=400, f_start=340e3, f_stop=360e3, f_step=5e3, g=20.0))
        if rc["A"] > 1e-30 and rf["A"] > 1e-30:
            assert abs(rc["A"] - rf["A"]) / rf["A"] < 0.10

class TestSHO:
    def test_keys(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=100, f_start=300e3, f_stop=400e3, f_step=2e3, g=20.0)
        d = sho_analysis(frequency_sweep(cant, contact, exc, opts))[0]
        for k in ("f_res", "A_res", "phase_res", "Q", "converged"):
            assert k in d

    def test_fres_in_range(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=100, f_start=300e3, f_stop=400e3, f_step=2e3, g=20.0)
        d = sho_analysis(frequency_sweep(cant, contact, exc, opts))[0]
        assert 300e3 <= d["f_res"] <= 400e3

class TestAnalyticalSolver:
    """Cross-validate the analytical solver against the BVP solver."""

    def _opts(self, method, n=150):
        return SolverOptions(n_points=n, f_start=300e3, f_stop=400e3,
                             f_step=5e3, g=20.0, method=method)

    def test_amplitude_agrees(self, base_params):
        cant, contact, exc, _ = base_params
        f = 350e3
        r_bvp  = solve_single_frequency(f, cant, contact, exc, self._opts('bvp'))
        r_anal = solve_single_frequency(f, cant, contact, exc, self._opts('analytical'))
        # Amplitudes should agree to within 1 %
        assert abs(r_bvp['A'] - r_anal['A']) / max(r_bvp['A'], 1e-30) < 0.01

    def test_phase_agrees(self, base_params):
        cant, contact, exc, _ = base_params
        f = 350e3
        r_bvp  = solve_single_frequency(f, cant, contact, exc, self._opts('bvp'))
        r_anal = solve_single_frequency(f, cant, contact, exc, self._opts('analytical'))
        # Phase difference < 2 degrees
        d_phase = abs(r_bvp['phase'] - r_anal['phase'])
        assert min(d_phase, abs(d_phase - 2*np.pi)) < np.radians(2.0)

    def test_no_distributed_load(self, simple_cant):
        """Without distributed load the particular solution is zero; check agreement."""
        exc_nodist = ExcitationParams(d1=1e-12, d2=0.0, f0=1e-11,
                                      delta_V=0.0, V_AC=0.0, eps_r=1.0,
                                      use_distributed_load=False, use_f0=True)
        contact = ContactParams(k1=500.0, k2=425.0, c1=0.0, c2=0.0)
        f = 300e3
        r_bvp  = solve_single_frequency(f, simple_cant, contact, exc_nodist,
                                        self._opts('bvp', n=200))
        r_anal = solve_single_frequency(f, simple_cant, contact, exc_nodist,
                                        self._opts('analytical', n=200))
        assert abs(r_bvp['A'] - r_anal['A']) / max(r_bvp['A'], 1e-30) < 0.01

    def test_bc_satisfied_analytical(self, base_params):
        """Check z(0)=0 and z'(0)=0 are satisfied by the analytical solution."""
        cant, contact, exc, _ = base_params
        opts = self._opts('analytical', n=300)
        res = solve_single_frequency(350e3, cant, contact, exc, opts)
        assert abs(res['z'][0])  < 1e-20
        assert abs(res['zp'][0]) < 1e-20

    def test_frequency_sweep_analytical(self, base_params):
        """Analytical sweep produces a valid resonance peak in the expected range."""
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=200, f_start=300e3, f_stop=400e3,
                             f_step=2e3, g=20.0, method='analytical')
        sweep = frequency_sweep(cant, contact, exc, opts)
        d = sho_analysis(sweep)[0]
        assert 300e3 <= d['f_res'] <= 400e3

    def test_analytical_bc_diagnostics_present(self, base_params):
        cant, contact, exc, _ = base_params
        res = solve_single_frequency(
            350e3, cant, contact, exc,
            SolverOptions(n_points=200, f_start=300e3, f_stop=400e3,
                          f_step=2e3, g=20.0, method='analytical')
        )
        diag = res["analytical_bc_diagnostics"]
        for key in ("freq_hz", "det_abs", "cond_raw",
                    "cond_row_scaled", "cond_rowcol_scaled",
                    "residual_raw", "solution_rel_diff_scaled_vs_raw"):
            assert key in diag


class TestKappaMapDefaults:
    def test_kappa_map_uses_contact_ratio_by_default(self):
        from eb_cr_afm import default_params_multi75
        from eb_cr_afm.spectra import kappa_map

        cant, contact, _, opts = default_params_multi75()
        km_default = kappa_map(cant, contact, opts, n_kappa=80, n_modes=2)
        km_ratio = kappa_map(cant, contact, opts, n_kappa=80, n_modes=2,
                             k2_over_k1=contact.k2 / contact.k1)

        assert km_default["kappa_zero_m2"] == pytest.approx(km_ratio["kappa_zero_m2"])


class TestConditioningSweep:
    def test_conditioning_sweep_collects_metrics(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=120, f_start=340e3, f_stop=360e3,
                             f_step=5e3, g=20.0, method='analytical')
        sweep = analytical_conditioning_sweep(cant, contact, exc, opts)

        assert len(sweep["freqs"]) == 5
        assert sweep["success"].all()
        assert np.all(np.isfinite(sweep["cond_raw"]))
        assert np.all(np.isfinite(sweep["cond_row_scaled"]))
        assert np.all(np.isfinite(sweep["cond_rowcol_scaled"]))
        assert np.all(np.isfinite(sweep["det_abs"]))

    def test_conditioning_sweep_requires_analytical_method(self, base_params):
        cant, contact, exc, _ = base_params
        opts = SolverOptions(n_points=100, f_start=340e3, f_stop=360e3,
                             f_step=5e3, g=20.0, method='bvp')
        with pytest.raises(ValueError):
            analytical_conditioning_sweep(cant, contact, exc, opts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
