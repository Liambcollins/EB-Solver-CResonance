"""
Spectral analysis: 2D maps, contact-stiffness sweeps, shape factors.

References
----------
Bradler et al., J. Appl. Phys. 120, 165107 (2016)
Bradler et al., J. Appl. Phys. 122, 065106 (2017)
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from .params import CantileverParams, ContactParams, ExcitationParams, SolverOptions
from .solver import frequency_sweep, solve_single_frequency, det_bc_matrix, mode_shape_at_tip


# numpy >= 2.0 renamed trapz -> trapezoid (trapz was later removed).
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def compute_2d_spectra(cant, contact, exc, opts, n_x_points=100, verbose=False):
    """
    Compute z(x,f) and z'(x,f) amplitude/phase maps.

    Returns dict: freqs, x_grid, z_amp, z_phase, zp_amp, zp_phase,
                  tip_amp, tip_phase
    """
    from scipy.interpolate import interp1d
    freqs = opts.frequencies
    Nf = len(freqs)
    x_out = np.linspace(0.0, cant.L, n_x_points)

    z_amp   = np.zeros((n_x_points, Nf))
    z_phase = np.zeros((n_x_points, Nf))
    zp_amp   = np.zeros((n_x_points, Nf))
    zp_phase = np.zeros((n_x_points, Nf))
    tip_amp  = np.zeros(Nf)
    tip_phase = np.zeros(Nf)

    for i, f in enumerate(freqs):
        if verbose and i % 20 == 0:
            print(f"  2D spectra: f={f/1e3:.1f} kHz  ({i+1}/{Nf})")
        res = solve_single_frequency(f, cant, contact, exc, opts)
        x_s  = res['x']
        z_s  = res['z']
        zp_s = res['zp']

        def _interp_mag(vals):
            return interp1d(x_s, np.abs(vals), kind='linear',
                            fill_value='extrapolate')(x_out)
        def _interp_ph(vals):
            return interp1d(x_s,
                            np.degrees(np.arctan2(np.imag(vals), np.real(vals))),
                            kind='linear', fill_value='extrapolate')(x_out)

        z_amp[:,i]   = _interp_mag(z_s)
        z_phase[:,i] = _interp_ph(z_s)
        zp_amp[:,i]  = _interp_mag(zp_s)
        zp_phase[:,i]= _interp_ph(zp_s)
        tip_amp[i]   = res['A']
        tip_phase[i] = np.degrees(res['phase'])

    return dict(freqs=freqs, x_grid=x_out,
                z_amp=z_amp, z_phase=z_phase,
                zp_amp=zp_amp, zp_phase=zp_phase,
                tip_amp=tip_amp, tip_phase=tip_phase)


def contact_stiffness_sweep(k1_values, cant, contact, exc, opts,
                             k2_over_k1=0.85, c1_value=None, verbose=False):
    """
    Sweep k1 over k1_values, finding all resonance modes in the frequency window.

    Each frequency sweep is analysed with sho_analysis: every peak is detected
    and fitted to an SHO model, so multiple eigenmodes are captured if they fall
    within opts.frequencies.

    Returns
    -------
    dict with keys:
        'k1'       : k1_values array
        'f_res'    : dominant-mode resonance frequency per k1 [Hz]  (backward compat)
        'A_res'    : dominant-mode peak amplitude per k1             (backward compat)
        'Q'        : dominant-mode quality factor per k1             (backward compat)
        'all_modes': list (len = n_k1) of lists of mode dicts, each dict having
                     keys 'f_res', 'A_res', 'Q', 'phase_res', 'converged'.
                     Modes within each k1 are sorted by ascending frequency
                     (mode 1 = lowest, mode 2 = next, …).
    """
    from .solver import sho_analysis as _sho
    f_res_arr = np.zeros(len(k1_values))
    A_res_arr = np.zeros(len(k1_values))
    Q_arr     = np.zeros(len(k1_values))
    all_modes = []

    for i, k1 in enumerate(k1_values):
        if verbose:
            print(f"  k1 sweep: k1={k1:.1f} N/m  ({i+1}/{len(k1_values)})")
        c_use = c1_value if c1_value is not None else contact.c1
        mod_c = ContactParams(
            k1=k1, k2=k2_over_k1*k1,
            c1=c_use, c2=k2_over_k1*c_use,
            psi=contact.psi)
        sweep = frequency_sweep(cant, mod_c, exc, opts)
        modes = _sho(sweep)
        # Sort by frequency (ascending) so index 0 = lowest mode
        modes_by_freq = sorted(modes, key=lambda m: m['f_res'])
        all_modes.append(modes_by_freq)

        if modes:
            dominant = max(modes, key=lambda m: m['A_res'])
            f_res_arr[i] = dominant['f_res']
            A_res_arr[i] = dominant['A_res']
            Q_arr[i]     = dominant['Q']
        else:
            f_res_arr[i] = np.nan
            A_res_arr[i] = np.nan
            Q_arr[i]     = np.nan

    return dict(k1=k1_values, f_res=f_res_arr, A_res=A_res_arr, Q=Q_arr,
                all_modes=all_modes)


def compute_shape_factor(res, cant):
    """
    Shape factor lambda = S_dynamic / S_static  (Paper 2, Eq. 7).
    """
    x  = res['x']
    z  = res['z']
    zp = res['zp']
    L  = x[-1]
    z_L  = z[-1]
    zp_L = zp[-1]
    z_sq_mean = _trapezoid(np.abs(z)**2, x) / L
    if np.abs(z_sq_mean) < 1e-60:
        return float('nan')
    S_static = 3.0 / (2.0 * L * np.cos(cant.phi))
    lambda_val = (1.0/S_static) * np.abs(z_L)*np.abs(zp_L) / z_sq_mean
    return float(lambda_val)


def kappa_map(cant: CantileverParams,
              contact_base: ContactParams,
              opts: SolverOptions,
              kappa_max: float = 600.0,
              n_kappa: int = 80,
              kappa_values: Optional[np.ndarray] = None,
              n_modes: int = 2,
              k2_over_k1: Optional[float] = None,
              n_scan: int = 600,
              n_refine: int = 200,
              verbose: bool = False,
              _progress_fn=None) -> dict:
    """
    Resonance frequency and mode-shape observables vs κ = k₁ L³ / EI.

    Uses eigenfrequency tracking (det_bc_matrix minimisation) so that both
    Mode 1 and Mode 2 are followed cleanly through the z'(L)=0 zero-crossings
    at κ ≈ 28.4  (Mode 1, λ → π)
    at κ ≈ 249   (Mode 2, λ → 2π).

    The physical message: at these special κ values the cantilever is
    *vibrating* (eigenfrequency is well-defined), but the OBD photodetector
    signal |z'(L)| vanishes.  |z(L)| (tip displacement) remains nonzero.

    Parameters
    ----------
    cant : CantileverParams
    contact_base : ContactParams
        k1 is overridden at each κ point.  c1, c2, psi are preserved for
        the driven-response evaluation.  Set to zero for the mode-shape
        (undamped) calculation.
    opts : SolverOptions
        g and n_points are used; frequency range is overridden internally.
    kappa_max : float
        Upper κ limit (default 600).
    n_kappa : int
        Number of linearly-spaced κ points.
    kappa_values : array-like, optional
        Override the κ grid entirely.
    n_modes : int
        1 or 2 modes to track (default 2).
    k2_over_k1 : float, optional
        k2 = k2_over_k1 * k1 at each κ point. If omitted, infer the ratio from
        ``contact_base``; if that is not possible, fall back to 0.85.
    n_scan, n_refine : int
        Coarse / fine scan grid sizes for det_bc_matrix minimisation.
    verbose : bool

    Returns
    -------
    dict with keys:
        'kappa'         : κ array, shape (n_kappa,)
        'k1'            : k1 array [N/m]
        'f_res'         : list of n_modes arrays — eigenfrequency [Hz]
        'zp_norm'       : list of n_modes arrays — |z'(L)| normalised to κ=0
        'zL_norm'       : list of n_modes arrays — |z(L)|  normalised to κ=0
        'f_free'        : list of n_modes free-air resonance frequencies [Hz]
        'f_pinned'      : list of n_modes pinned-limit frequencies [Hz]
        'kappa_zero_m1' : κ of Mode 1 z'(L) minimum (~28.4 for simplified model)
        'kappa_zero_m2' : κ of Mode 2 z'(L) minimum (~249)  [nan if n_modes<2]
        'EI', 'L', 'm'  : cantilever parameters used
    """
    EI = cant.EI
    L  = cant.contact_x
    m  = cant.m

    # Free-air and pinned-limit eigenvalue multipliers (analytic, no tip mass)
    FREE_LAMBDAS   = [1.8751, 4.6941, 7.8548][:n_modes]
    PINNED_LAMBDAS = [3.9274, 7.0686, 10.2102][:n_modes]
    f_scale  = np.sqrt(EI / (m * L**3)) / (2.0 * np.pi)
    f_free   = [lam**2 * f_scale for lam in FREE_LAMBDAS]
    f_pinned = [lam**2 * f_scale for lam in PINNED_LAMBDAS]

    if kappa_values is None:
        kappa_values = np.linspace(0.0, kappa_max, n_kappa)
    kappa_values = np.asarray(kappa_values, dtype=float)

    if k2_over_k1 is None:
        if abs(contact_base.k1) > 1e-30:
            k2_over_k1 = float(contact_base.k2 / contact_base.k1)
        else:
            k2_over_k1 = 0.85

    # SolverOptions for eigenfrequency search: zero damping → sharp det minima
    from .params import SolverOptions as SO
    opts_eig = SO(n_points=opts.n_points, f_start=1.0, f_stop=1.0,
                  f_step=1.0, g=0.0, method='analytical')

    def _find_eig(contact_nd: ContactParams, f_lo: float, f_hi: float) -> float:
        """Locate eigenfrequency via |det(M)| minimisation in [f_lo, f_hi]."""
        f_c = np.linspace(f_lo, f_hi, n_scan)
        d_c = np.array([det_bc_matrix(2.0*np.pi*f, cant, contact_nd, opts_eig)
                        for f in f_c])
        i   = int(np.argmin(d_c))
        lo2 = f_c[max(0, i - 2)];  hi2 = f_c[min(n_scan - 1, i + 2)]
        f_r = np.linspace(lo2, hi2, n_refine)
        d_r = np.array([det_bc_matrix(2.0*np.pi*f, cant, contact_nd, opts_eig)
                        for f in f_r])
        return float(f_r[np.argmin(d_r)])

    # Mode n search windows:
    #   Mode 1: [0.9 * f1_free, 1.05 * f1_pinned]
    #   Mode 2: must start above Mode 1 pinned limit to avoid confusion
    f1_lo = 0.90 * f_free[0];    f1_hi = 1.05 * f_pinned[0]
    f2_lo = max(0.97 * f_free[1], 1.02 * f_pinned[0]) if n_modes >= 2 else None
    f2_hi = 1.05 * f_pinned[1]                         if n_modes >= 2 else None

    f_res_arr  = [np.zeros(len(kappa_values)) for _ in range(n_modes)]
    zp_arr     = [np.zeros(len(kappa_values)) for _ in range(n_modes)]
    zL_arr     = [np.zeros(len(kappa_values)) for _ in range(n_modes)]

    for ik, kappa in enumerate(kappa_values):
        k1 = kappa * EI / L**3
        k2 = k2_over_k1 * k1
        if verbose:
            print(f"  kappa={kappa:.1f}  k1={k1:.4f} N/m  ({ik+1}/{len(kappa_values)})")

        # Undamped contact (for eigenfrequency + mode shape)
        c_nd = ContactParams(k1=max(k1, 1e-9), k2=k2, c1=0.0, c2=0.0,
                             psi=contact_base.psi)

        # Mode 1
        f1_eig = _find_eig(c_nd, f1_lo, f1_hi)
        f_res_arr[0][ik] = f1_eig
        z_L1, zp_L1 = mode_shape_at_tip(2.0*np.pi*f1_eig, cant, c_nd, opts_eig)
        zL_arr[0][ik]  = abs(z_L1)
        zp_arr[0][ik]  = abs(zp_L1)

        # Mode 2 (if requested)
        if n_modes >= 2:
            f2_eig = _find_eig(c_nd, f2_lo, f2_hi)
            f_res_arr[1][ik] = f2_eig
            z_L2, zp_L2 = mode_shape_at_tip(2.0*np.pi*f2_eig, cant, c_nd, opts_eig)
            zL_arr[1][ik]  = abs(z_L2)
            zp_arr[1][ik]  = abs(zp_L2)

        # Progress callback for GUI integration
        if _progress_fn is not None:
            _progress_fn(ik, len(kappa_values))

    # Normalise each mode to its κ=0 value
    def _safe_norm(arr: np.ndarray) -> np.ndarray:
        v0 = arr[0]
        return arr / v0 if abs(v0) > 1e-40 else arr

    zp_norm = [_safe_norm(a) for a in zp_arr]
    zL_norm = [_safe_norm(a) for a in zL_arr]

    def _zero_kappa(norm_arr: np.ndarray) -> float:
        """κ of the minimum of norm_arr (≈ zero-crossing if min < 0.01)."""
        mn = int(np.argmin(norm_arr))
        return float(kappa_values[mn]) if norm_arr[mn] < 0.05 else float('nan')

    return dict(
        kappa=kappa_values,
        k1=kappa_values * EI / L**3,
        f_res=f_res_arr,
        zp_norm=zp_norm,
        zL_norm=zL_norm,
        zp_raw=zp_arr,
        zL_raw=zL_arr,
        f_free=f_free,
        f_pinned=f_pinned,
        f_scale=f_scale,
        kappa_zero_m1=_zero_kappa(zp_norm[0]) if n_modes >= 1 else float('nan'),
        kappa_zero_m2=_zero_kappa(zp_norm[1]) if n_modes >= 2 else float('nan'),
        EI=EI, L=L, m=m,
    )


def compute_stiffness_frequency_map(cant, contact_base, exc, opts,
                                     k1_values=None, n_k1=30,
                                     k1_min=None, k1_max=None,
                                     k2_over_k1=0.85, verbose=False):
    import numpy as np
    from .params import ContactParams
    from .solver import solve_single_frequency

    freqs = opts.frequencies
    Nf = len(freqs)

    if k1_values is None:
        _k1_min = k1_min if k1_min is not None else 0.01 * cant.k
        _k1_max = k1_max if k1_max is not None else 5000.0 * cant.k
        k1_values = np.logspace(np.log10(_k1_min), np.log10(_k1_max), n_k1)

    Nk = len(k1_values)
    amp_map   = np.zeros((Nk, Nf))
    phase_map = np.zeros((Nk, Nf))

    for ik, k1 in enumerate(k1_values):
        if verbose:
            print(f"  k1={k1:.2f} N/m  ({ik+1}/{Nk})")
        mod_contact = ContactParams(
            k1=k1, k2=k2_over_k1*k1,
            c1=contact_base.c1, c2=k2_over_k1*contact_base.c1,
            psi=contact_base.psi)
        for jf, f in enumerate(freqs):
            res = solve_single_frequency(f, cant, mod_contact, exc, opts)
            amp_map[ik, jf]   = res["A"]
            phase_map[ik, jf] = np.degrees(res["phase"])

    return dict(freqs=freqs, k1=k1_values, amp_map=amp_map, phase_map=phase_map)
