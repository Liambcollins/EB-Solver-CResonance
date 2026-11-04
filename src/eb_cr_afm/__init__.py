"""
eb_cr_afm - Euler-Bernoulli Contact-Resonance AFM Solver

Based on Bradler et al., J. Appl. Phys. 120, 165107 (2016) and 122, 065106
(2017), extended with a two-segment model in which the tip is set back from
the free end of the lever.  If you use this package, please cite

    M. Checa et al., "Dynamics of Null and Electrostatic Blind Spots for
    Quantitative PFM", ACS Nano (2026); arXiv:2607.27486.
"""
import numpy as np

from .params import CantileverParams, ContactParams, ExcitationParams, SolverOptions
from .solver import (solve_single_frequency, frequency_sweep,
    analytical_conditioning_sweep,
    sho_analysis,
    compute_alpha4, compute_contact_stiffnesses, compute_auxiliary,
    compute_tip_forces, distributed_load,
    det_bc_matrix, mode_shape_at_tip)
from .spectra import (compute_2d_spectra, contact_stiffness_sweep,
    compute_shape_factor, kappa_map)
from .plotting import (plot_transfer_function, plot_displacement_profile,
    plot_2d_spectra, plot_stiffness_sweep, plot_sho_check,
    plot_modes_and_spectra, plot_kappa_map)
from .interactive import (plot_transfer_function_plotly, plot_displacement_profile_plotly,
    plot_stiffness_sweep_plotly, plot_stiffness_map_plotly)

__version__ = "0.1.0"


def default_params():
    """Return Table I parameters from Bradler et al. (2016)."""
    cant = CantileverParams(L=240e-6, w=40e-6, H=6e-6, k=3.0,
                            m=5e-11, mT=5e-12, phi=0.2)
    contact = ContactParams(k1=1000.0, k2=850.0, c1=20e-6, c2=17e-6, psi=0.0)
    exc = ExcitationParams(d1=1e-12, d2=0.0, f0=100e-12,
                           delta_V=1.0, V_AC=1.0, eps_r=1.0,
                           use_distributed_load=True, use_f0=True)
    opts = SolverOptions(n_points=300, f_start=300e3, f_stop=400e3,
                         f_step=500.0, g=20000.0)   # Table I: eta = 20 x 1000/s
    return cant, contact, exc, opts


def default_params_multi75():
    """Return typical parameters for a Budget Sensors Multi 75-G probe.

    Cantilever geometry (Multi 75-G nominal):
        L = 225 µm,  w = 28 µm,  H = 12 µm,  phi = 11°
        k = 2.8 N/m  (nominal spring constant)
        m ≈ 5.2e-11 kg  (back-calculated from f_free ≈ 75 kHz)

    The mass m is derived so that the first free-cantilever resonance
    satisfies ω = (1.875/L)^2 * sqrt(EI·L / m) ≈ 2π·75 kHz.

    Contact defaults assume a moderately stiff sample:
        k1 = 100 N/m (k1/k ≈ 36),  k2 = 0.85 * k1

    Sweep covers the full first contact-resonance mode transition
    from the free resonance (~75 kHz) to the clamped-pinned limit (~329 kHz).
    """
    k = 2.8            # N/m
    L = 225e-6         # m
    # Mass that gives f_free = 75 kHz for this L and k:
    #   omega = beta1^2 * sqrt(EI*L / m),  beta1 = 1.875/L
    #   => m = beta1^4 * EI * L / omega_free^2
    EI = k * L**3 / 3
    omega_free = 2 * np.pi * 75e3
    beta1 = 1.875 / L
    m = beta1**4 * EI * L / omega_free**2

    cant = CantileverParams(
        L=L,
        w=28e-6,
        H=12e-6,
        k=k,
        m=m,
        mT=0.0,
        phi=np.radians(11.0),   # 11° nominal tilt
    )
    contact = ContactParams(k1=100.0, k2=85.0, c1=0.0, c2=0.0, psi=0.0)
    exc = ExcitationParams(
        d1=1e-12,
        d2=0.0,
        f0=0.0,
        use_distributed_load=False,
        use_f0=False,
    )
    opts = SolverOptions(
        n_points=300,
        f_start=50e3,
        f_stop=400e3,
        f_step=500.0,
        g=2000.0,   # eta ~ 2e3 1/s, typical soft-lever air damping (P2 Table I)
    )
    return cant, contact, exc, opts


def default_params_null_spot_paper():
    """Return the Multi75E-G parameter set used in Checa et al. (ACS Nano 2026).

    BudgetSensors Multi75E-G geometry with the tip set back
    ``tip_offset = 10.9 um`` from the free end (measured by SEM), so the
    two-segment overhang model is active.  The contact is a moderately stiff
    k1 = 1000 N/m with a compliance-limited lateral spring
    (k1 in series with a 2 kN/m tip cone, k2 ~ 667 N/m).  Both drives are on:
    a 3 pm vertical piezo drive and the distributed electrostatic load
    (V_AC = 3 V, dV = 1 V), which is the situation where null spots and
    electrostatic blind spots matter.

    See examples/10_manuscript_figures.ipynb for the full figure set.
    """
    cant = CantileverParams(
        L=225e-6, w=28e-6, H=12e-6, k=2.8, m=5.2e-11, mT=0.0,
        phi=np.radians(11.0), tip_offset=10.9e-6,
    )
    k1 = 1000.0
    k_cone = 2000.0
    contact = ContactParams(k1=k1, k2=1.0 / (1.0 / k1 + 1.0 / k_cone),
                            c1=20e-6, c2=0.0, psi=0.0)
    exc = ExcitationParams(
        d1=3e-12, d2=0.0, f0=0.0,
        delta_V=1.0, V_AC=3.0, eps_r=1.0,
        use_distributed_load=True, use_f0=False,
    )
    opts = SolverOptions(n_points=400, f_start=30e3, f_stop=700e3,
                         f_step=250.0, g=100.0)
    return cant, contact, exc, opts


__all__ = [
    "CantileverParams", "ContactParams", "ExcitationParams", "SolverOptions",
    "solve_single_frequency", "frequency_sweep", "analytical_conditioning_sweep", "sho_analysis",
    "compute_alpha4", "compute_contact_stiffnesses", "compute_auxiliary",
    "compute_tip_forces", "distributed_load",
    "compute_2d_spectra", "contact_stiffness_sweep", "compute_shape_factor",
    "kappa_map",
    "plot_transfer_function", "plot_displacement_profile",
    "plot_2d_spectra", "plot_stiffness_sweep", "plot_sho_check",
    "plot_modes_and_spectra", "plot_kappa_map",
    "det_bc_matrix", "mode_shape_at_tip",
    "plot_transfer_function_plotly", "plot_displacement_profile_plotly",
    "plot_stiffness_sweep_plotly", "plot_stiffness_map_plotly",
    "default_params", "default_params_multi75", "default_params_null_spot_paper",
]
