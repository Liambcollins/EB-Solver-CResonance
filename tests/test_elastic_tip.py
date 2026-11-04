"""
Tests for the elastic-tip (compliant cone) contact extension:
ContactParams.k_cone_lat / k_cone_ax and solver.compute_effective_contact.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eb_cr_afm import (CantileverParams, ContactParams, ExcitationParams,
                       SolverOptions, solve_single_frequency)
from eb_cr_afm.solver import (compute_auxiliary, compute_effective_contact)


@pytest.fixture
def cant():
    return CantileverParams(L=225e-6, w=28e-6, H=12e-6, k=2.8, m=5.2e-11,
                            mT=0.0, phi=np.radians(11.0), tip_offset=15e-6)


def exc_es():
    return ExcitationParams(d1=0, d2=0, f0=0, use_f0=False, bias_coupled_f0=False,
                            delta_V=1.0, V_AC=1.0, eps_r=1.0,
                            use_distributed_load=True)


def test_rigid_limit_matches_compute_auxiliary():
    """C_tip = 0 must reproduce Bradler's T, X, U and unfiltered drive forces
    for random stiffness/damping/angle combinations."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        cant = CantileverParams(L=225e-6, w=28e-6, H=rng.uniform(3, 30) * 1e-6,
                                k=2.8, m=5.2e-11, mT=rng.uniform(0, 1e-11),
                                phi=rng.uniform(-0.4, 0.4), tip_offset=15e-6)
        ct = ContactParams(k1=rng.uniform(1e2, 1e6), k2=rng.uniform(0, 1e6),
                           c1=rng.uniform(0, 1e-4), c2=rng.uniform(0, 1e-4),
                           psi=rng.uniform(-0.2, 0.2))
        omega = rng.uniform(1e5, 5e6)
        a4 = (cant.m / (cant.EI * cant.L)) * (omega**2 - 1j * 20.0 * omega)
        k1c = ct.k1 + 1j * omega * ct.c1
        k2c = ct.k2 + 1j * omega * ct.c2
        T0, X0, U0 = compute_auxiliary(a4, k1c, k2c, cant, ct)
        Fv0 = k1c * 3e-12 + (0.3 + 0.1j) * 1e-10
        Fl0 = k2c * 1e-12
        T1, X1, U1, Fv1, Fl1 = compute_effective_contact(
            a4, k1c, k2c, cant, ct, Fv0, Fl0)
        for a, b in ((T0, T1), (X0, X1), (U0, U1), (Fv0, Fv1), (Fl0, Fl1)):
            assert abs(a - b) <= 1e-10 * max(abs(a), 1e-300)


def test_infinite_cone_is_bitwise_inert(cant):
    """Default (inf) cone must leave solve_single_frequency untouched."""
    opts = SolverOptions(n_points=400, g=20.0, method="analytical")
    ct_rigid = ContactParams(k1=1e5, k2=1e5, c1=20e-6, c2=0.0)
    ct_inf = ContactParams(k1=1e5, k2=1e5, c1=20e-6, c2=0.0,
                           k_cone_lat=np.inf, k_cone_ax=np.inf)
    for f in (30e3, 450e3):
        r0 = solve_single_frequency(f, cant, ct_rigid, exc_es(), opts)
        r1 = solve_single_frequency(f, cant, ct_inf, exc_es(), opts)
        assert np.array_equal(r0["z"], r1["z"])


def test_very_stiff_cone_approaches_rigid(cant):
    opts = SolverOptions(n_points=400, g=20.0, method="analytical")
    ct_rigid = ContactParams(k1=1e5, k2=1e5, c1=20e-6, c2=0.0)
    ct_stiff = ContactParams(k1=1e5, k2=1e5, c1=20e-6, c2=0.0,
                             k_cone_lat=1e13, k_cone_ax=1e15)
    for f in (30e3, 450e3):
        r0 = solve_single_frequency(f, cant, ct_rigid, exc_es(), opts)
        r1 = solve_single_frequency(f, cant, ct_stiff, exc_es(), opts)
        dev = np.max(np.abs(r0["z"] - r1["z"])) / np.max(np.abs(r0["z"]))
        assert dev < 1e-6


def test_analytical_vs_bvp_with_finite_cone(cant):
    """The elastic tip enters upstream of both solver paths; they must agree."""
    ct = ContactParams(k1=1e5, k2=1e5, c1=20e-6, c2=0.0, k_cone_lat=2000.0)
    oa = SolverOptions(n_points=600, g=20.0, method="analytical")
    ob = SolverOptions(n_points=600, g=20.0, method="bvp")
    for f in (30e3, 460e3):
        ra = solve_single_frequency(f, cant, ct, exc_es(), oa)
        rb = solve_single_frequency(f, cant, ct, exc_es(), ob)
        za = (np.interp(rb["x"], ra["x"], np.real(ra["z"]))
              + 1j * np.interp(rb["x"], ra["x"], np.imag(ra["z"])))
        dev = np.max(np.abs(za - rb["z"])) / np.max(np.abs(rb["z"]))
        assert dev < 1e-3


def test_series_softening_of_rotational_constraint(cant):
    """A finite lateral cone stiffness must cap the effective rotational
    constraint: K_eff_xx (the H**2 term in T) saturates at ~k_cone_lat for
    k2 >> k_cone_lat, instead of growing with k2."""
    omega = 2 * np.pi * 4.6e5
    a4 = (cant.m / (cant.EI * cant.L)) * (omega**2 - 1j * 20.0 * omega)
    ct_soft = ContactParams(k1=1e6, k2=1e6, c1=0.0, c2=0.0, k_cone_lat=2000.0)
    T, X, U, _, _ = compute_effective_contact(a4, 1e6 + 0j, 1e6 + 0j,
                                              cant, ct_soft, 0j, 0j)
    K_xx_eff = np.real(T) * cant.EI / cant.H**2
    assert K_xx_eff < 1.2 * 2000.0     # saturated at the cone stiffness
    assert K_xx_eff > 0.5 * 2000.0


def test_elastic_tip_releases_kinematic_lock(cant):
    """With a rigid tip and stiff isotropic contact, resonant (piezo) and
    quasistatic (electrostatic) displacement nulls coincide with drive-
    independent positions; the elastic cone makes them drive-dependent with
    the D-ESBS on the tip side of the D-NS (the FEM ordering)."""
    from scipy.interpolate import CubicSpline

    def null_x(r):
        x = r["x"]
        m = x >= cant.L - 70e-6
        xs = np.linspace(x[m][0], x[-1], 20001)
        A = np.abs(CubicSpline(x, np.real(r["z"]))(xs)
                   + 1j * CubicSpline(x, np.imag(r["z"]))(xs))
        return xs[int(np.argmin(A))]

    opts = SolverOptions(n_points=800, g=20.0, method="analytical")
    exc_pz = ExcitationParams(d1=3e-12, d2=0, f0=0, use_f0=False,
                              bias_coupled_f0=False, delta_V=0, V_AC=0,
                              use_distributed_load=False)
    kstar = 1e4
    ct = ContactParams(k1=kstar, k2=kstar, c1=0.0, c2=0.0, k_cone_lat=2000.0)
    # resonance near 447 kHz for this configuration (coarse scan)
    freqs = np.linspace(430e3, 465e3, 36)
    amps = [solve_single_frequency(float(f), cant, ct, exc_pz, opts)["A_disp"]
            for f in freqs]
    fcr = float(freqs[int(np.argmax(amps))])
    x_dns = null_x(solve_single_frequency(fcr, cant, ct, exc_pz, opts))
    x_esbs = null_x(solve_single_frequency(30e3, cant, ct, exc_es(), opts))
    sep_um = (x_esbs - x_dns) * 1e6
    assert sep_um > 1.0        # drive-dependent, tip-side D-ESBS
    assert x_esbs > x_dns      # FEM ordering
