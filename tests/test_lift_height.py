"""
Tests for ExcitationParams.lift_height and its effect on distributed_load().
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eb_cr_afm import CantileverParams, ContactParams, ExcitationParams, SolverOptions
from eb_cr_afm import solve_single_frequency
from eb_cr_afm.solver import distributed_load

epsilon_0 = 8.854187817e-12  # F/m


@pytest.fixture
def cant():
    return CantileverParams(L=240e-6, w=40e-6, H=6e-6, k=3.0,
                            m=5e-11, mT=5e-12, phi=0.2)


@pytest.fixture
def x_grid(cant):
    return np.linspace(0.0, cant.contact_x, 200)


@pytest.fixture
def exc_base():
    return ExcitationParams(
        d1=1e-12, d2=0.0, f0=0.0,
        delta_V=1.0, V_AC=1.0, eps_r=1.0,
        use_distributed_load=True, use_f0=False,
        lift_height=0.0,
    )


class TestLiftHeightDefault:
    """Test 1 — Default lift_height=0.0 is backward-compatible."""

    def test_zero_lift_matches_original_formula(self, cant, x_grid, exc_base):
        d_ref = (cant.H * np.cos(cant.phi)
                 + (cant.contact_x - x_grid) * np.sin(cant.phi))
        q_ref = (-np.cos(cant.phi) * epsilon_0 * exc_base.eps_r
                 * exc_base.V_AC * exc_base.delta_V * cant.w / d_ref**2)

        q_out = distributed_load(x_grid, cant, exc_base)

        # rtol=1e-9: adding exc.lift_height (== 0.0) to the d_x expression
        # changes the FP evaluation order, producing ~2e-10 relative rounding.
        np.testing.assert_allclose(q_out, q_ref.astype(complex), rtol=1e-9)


class TestLiftHeightMonotonic:
    """Test 2 — Larger lift_height → strictly smaller force magnitude."""

    def test_force_decreases_monotonically_with_lift(self, cant, x_grid):
        lift_values = [0.0, 10e-9, 25e-9, 50e-9]
        mean_forces = []
        for h in lift_values:
            exc = ExcitationParams(
                d1=0.0, d2=0.0, f0=0.0,
                delta_V=1.0, V_AC=1.0, eps_r=1.0,
                use_distributed_load=True, use_f0=False,
                lift_height=h,
            )
            q = distributed_load(x_grid, cant, exc)
            mean_forces.append(np.abs(q).mean())

        for i in range(len(mean_forces) - 1):
            assert mean_forces[i] > mean_forces[i + 1], (
                f"Force did not decrease: lift={lift_values[i]:.0e} -> "
                f"{lift_values[i+1]:.0e}, |q|={mean_forces[i]:.3e} -> {mean_forces[i+1]:.3e}"
            )


class TestLiftHeightNegativeGap:
    """Test 3 — Non-positive gap raises ValueError."""

    def test_negative_lift_raises(self, cant, x_grid):
        # Force d(x) < 0 by making lift_height strongly negative
        exc = ExcitationParams(
            d1=0.0, d2=0.0, f0=0.0,
            delta_V=1.0, V_AC=1.0, eps_r=1.0,
            use_distributed_load=True, use_f0=False,
            lift_height=-(cant.H + 1e-6),
        )
        with pytest.raises(ValueError, match="d\\(x\\) <= 0"):
            distributed_load(x_grid, cant, exc)

    def test_error_message_contains_lift_height(self, cant, x_grid):
        bad_lift = -(cant.H + 1e-6)
        exc = ExcitationParams(
            d1=0.0, d2=0.0, f0=0.0,
            delta_V=1.0, V_AC=1.0, eps_r=1.0,
            use_distributed_load=True, use_f0=False,
            lift_height=bad_lift,
        )
        with pytest.raises(ValueError, match="lift_height="):
            distributed_load(x_grid, cant, exc)


class TestLiftHeightEndToEnd:
    """Test 4 — solve_single_frequency propagates lift_height to body force."""

    def test_lift_reduces_response_amplitude(self, cant):
        contact = ContactParams(k1=3000.0, k2=0.85 * 3000.0, c1=20e-6, c2=0.0)
        opts = SolverOptions(n_points=300, f_start=300e3, f_stop=400e3,
                             f_step=2e3, g=20.0, method='analytical')

        exc_no_lift = ExcitationParams(
            d1=0.0, d2=0.0, f0=0.0,
            delta_V=1.0, V_AC=1.0, eps_r=1.0,
            use_distributed_load=True, use_f0=False,
            lift_height=0.0,
        )
        exc_with_lift = ExcitationParams(
            d1=0.0, d2=0.0, f0=0.0,
            delta_V=1.0, V_AC=1.0, eps_r=1.0,
            use_distributed_load=True, use_f0=False,
            lift_height=50e-9,
        )

        res_no_lift   = solve_single_frequency(350e3, cant, contact, exc_no_lift,   opts)
        res_with_lift = solve_single_frequency(350e3, cant, contact, exc_with_lift, opts)

        # Larger gap → smaller body force → smaller slope amplitude
        assert res_with_lift["A_slope"] < res_no_lift["A_slope"], (
            f"Expected A_slope to decrease with lift: "
            f"no_lift={res_no_lift['A_slope']:.3e}, "
            f"with_lift={res_with_lift['A_slope']:.3e}"
        )
