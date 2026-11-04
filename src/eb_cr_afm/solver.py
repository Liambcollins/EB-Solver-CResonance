"""
Core Euler–Bernoulli contact-resonance solver.

Faithfully implements the model of Bradler et al., J. Appl. Phys. 120, 165107 (2016).

Equation numbers refer to that paper unless prefixed with "P2:" which denotes
Bradler et al., J. Appl. Phys. 122, 065106 (2017).

Mathematical summary
--------------------
Governing ODE (Eq. 1):
    d⁴z/dx⁴ = α⁴ z(x) + q(x) / (EI)

Complex wavenumber (Eq. 2):
    α⁴ = (m / (EI·L)) · (ω² − i g ω)

Distributed load (Eqs. 3–4):
    q(d) = −cos(φ) ε₀ εᵣ V_AC ΔV w / d(x)²
    d(x) = H cos(φ) + (L − x) sin(φ)

Contact stiffnesses (Eqs. 5–6):
    k1c = k1 + i ω c1
    k2c = k2 + i ω c2

Contact angle (Eq. 7):
    θ = φ + ψ

Auxiliary functions (Eqs. 8–10):
    T = (H²/EI)(k1c sin²θ + k2c cos²θ) − (H/4)² (mT/m) α⁴ L
    X = (H/EI) sinθ cosθ (k2c − k1c)
    U = (1/EI)(k1c cos²θ + k2c sin²θ) − (mT/m) α⁴ L

Tip forces (Eqs. 11–12):
    F_vert = k1c d1 + f0
    F_lon  = k2c d2

Boundary conditions (Eqs. 13–16, in the form actually implemented):
    z(0)   = 0
    z'(0)  = 0
    z''(L)  + T z'(L) + X z(L)  = (H/EI) cosθ F_lon − (H/EI) sinθ F_vert
    z'''(L) − U z(L)  − X z'(L) = −(1/EI) cosθ F_vert − (1/EI) sinθ F_lon

Observables (Eqs. 17–18):
    A   = |z'(L)|
    φ_m = −atan2(Im[z'(L)], Re[z'(L)])

Tip set back from the free end (tip_offset > 0)
-----------------------------------------------
Bradler's tip boundary conditions hold at the beam terminus and are only
valid when the contact coincides with the free end.  When
``cant.tip_offset > 0`` the contact sits at an interior point
x_c = L − tip_offset and the model becomes a two-segment beam (the standard
Rabe CR-AFM geometry): segment 1 on [0, x_c], clamped at 0; segment 2 (the
overhang) on [x_c, L], load-free with a genuine free end,
z''(L) = z'''(L) = 0.  The tip reaction enters as interior jump conditions
at x_c:

    z, z' continuous,
    [z''(x_c⁻) − z''(x_c⁺)]  + T z'(x_c) + X z(x_c)  = RHS_mom
    [z'''(x_c⁻) − z'''(x_c⁺)] − U z(x_c) − X z'(x_c) = RHS_shear

which reduce exactly to Eqs. 15–16 in the limit tip_offset → 0 (a
zero-length free overhang carries no moment or shear).  Both the analytical
and the BVP paths implement this two-segment model; see
_solve_analytical_two_segment and _solve_bvp_two_segment.
"""

from __future__ import annotations

import numpy as np
try:
    from scipy.integrate import solve_bvp
    from scipy.constants import epsilon_0
    _SCIPY_AVAILABLE = True
except (ImportError, AttributeError):
    solve_bvp = None
    epsilon_0 = 8.854187817e-12   # F/m — fallback value
    _SCIPY_AVAILABLE = False

from .params import CantileverParams, ContactParams, ExcitationParams, SolverOptions


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def compute_alpha4(omega: complex, cant: CantileverParams, opts: SolverOptions) -> complex:
    """
    Complex wavenumber to the 4th power (Eq. 2).

    α⁴ = (m / (EI · L)) · (ω² − i g ω)

    Parameters
    ----------
    omega : complex
        Angular frequency [rad/s].
    cant : CantileverParams
    opts : SolverOptions

    Returns
    -------
    complex
        α⁴  [1/m⁴]
    """
    return (cant.m / (cant.EI * cant.L)) * (omega**2 - 1j * opts.g * omega)


def compute_contact_stiffnesses(omega: float,
                                contact: ContactParams) -> tuple[complex, complex]:
    """
    Complex contact stiffnesses (Eqs. 5–6).

    k1c = k1 + i ω c1
    k2c = k2 + i ω c2
    """
    k1c = contact.k1 + 1j * omega * contact.c1
    k2c = contact.k2 + 1j * omega * contact.c2
    return k1c, k2c


def compute_auxiliary(alpha4: complex, k1c: complex, k2c: complex,
                      cant: CantileverParams, contact: ContactParams) -> tuple[complex, complex, complex]:
    """
    Auxiliary functions T, X, U (Eqs. 8–10).

    T = (H²/EI)(k1c sin²θ + k2c cos²θ) − (H/4)² (mT/m) α⁴ L
    X = (H/EI) sinθ cosθ (k2c − k1c)
    U = (1/EI)(k1c cos²θ + k2c sin²θ) − (mT/m) α⁴ L

    where θ = φ + ψ  (Eq. 7).
    """
    theta = cant.phi + contact.psi
    EI = cant.EI
    H = cant.H
    # The α⁴L products below must use the same L as in α⁴ = m/(EI·L)·(ω²−igω)
    # so that (mT/m)·α⁴·L = mT(ω²−igω)/EI — the tip-mass inertia term, which
    # is independent of where the tip sits along the beam (it must not be
    # scaled by contact_x when tip_offset > 0).
    L = cant.L
    mT_over_m = cant.mT / cant.m

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    T = (H**2 / EI) * (k1c * sin_th**2 + k2c * cos_th**2) \
        - (H / 4)**2 * mT_over_m * alpha4 * L

    X = (H / EI) * sin_th * cos_th * (k2c - k1c)

    U = (1.0 / EI) * (k1c * cos_th**2 + k2c * sin_th**2) \
        - mT_over_m * alpha4 * L

    return T, X, U


def compute_tip_forces(omega: float, k1c: complex, k2c: complex,
                       exc: ExcitationParams) -> tuple[complex, complex]:
    """
    Tip excitation forces (Eqs. 11–12).

    F_vert = k1c d1 + f0_eff
    F_lon  = k2c d2

    Optional local electrostatic force:

        f_bias = 0.5 * (dC/dz) * V_bias^2

    with V_bias = delta_V + V_AC.
    """
    f0_eff = exc.f0 if exc.use_f0 else 0.0
    if exc.bias_coupled_f0:
        V_bias = exc.delta_V + exc.V_AC
        f0_eff = f0_eff + 0.5 * exc.capacitance_gradient * V_bias**2

    F_vert = k1c * exc.d1 + f0_eff
    F_lon = k2c * exc.d2
    return F_vert, F_lon


def compute_effective_contact(alpha4: complex, k1c: complex, k2c: complex,
                              cant: CantileverParams, contact: ContactParams,
                              F_vert: complex, F_lon: complex
                              ) -> tuple[complex, complex, complex, complex, complex]:
    """
    Elastic-tip (compliant cone) effective contact reaction and drive forces.

    The Bradler model treats the tip as a rigid lever arm of height H, so the
    apex displacement is kinematically locked to the beam state q = (z, z')
    at the contact: u = A q with A = [[0, H], [1, 0]] in the beam (x, z)
    frame.  A real tip cone is elastic; with apex compliance
    C_tip = diag(1/k_cone_lat, 1/k_cone_ax) the apex force balance becomes

        u = A q + C_tip f,          f = K_c (d_vec − u) + F_ext

    (f = total apex load, K_c = k1c n̂n̂ᵀ + k2c t̂t̂ᵀ the contact stiffness
    matrix, d_vec the sample-side drive displacement, F_ext the direct apex
    force).  Eliminating u gives

        f = S (K_c d_vec + F_ext) − S K_c A q,     S = (I + K_c C_tip)⁻¹

    i.e. the beam sees an effective contact stiffness K_eff = S K_c (the
    matrix series combination of contact and cone) and drive forces filtered
    by the same S.  With C_tip = 0 this reduces exactly to the rigid-tip
    T, X, U of compute_auxiliary and the unfiltered F_vert / F_lon.

    Physically this captures the two roles of cone elasticity: (i) it
    saturates the rotational constraint (H² K_eff_xx) that the lateral
    contact spring can impose, and (ii) it releases the rigid-tip kinematic
    lock z = H·tanθ·z' under a stiff normal contact by a drive-dependent
    amount, allowing resonant and quasistatic displacement nulls to sit at
    different near-tip positions (cf. the persistent D-NS/D-ESBS separation
    of the 3-D finite-element model).

    Returns
    -------
    (T, X, U, F_vert_eff, F_lon_eff) : all complex
        Drop-in replacements for the outputs of compute_auxiliary and
        compute_tip_forces.
    """
    theta = cant.phi + contact.psi
    sn, cs = np.sin(theta), np.cos(theta)
    # Contact frame in beam (x, z) coordinates, chosen so that the rigid
    # limit reproduces Bradler's T, X, U and RHS sign conventions exactly.
    n_hat = np.array([-sn, cs])   # contact normal
    t_hat = np.array([cs, sn])    # contact tangent
    Kc = (k1c * np.outer(n_hat, n_hat)
          + k2c * np.outer(t_hat, t_hat)).astype(complex)

    c_lat = 1.0 / contact.k_cone_lat if np.isfinite(contact.k_cone_lat) else 0.0
    c_ax = 1.0 / contact.k_cone_ax if np.isfinite(contact.k_cone_ax) else 0.0
    C_tip = np.diag([c_lat, c_ax]).astype(complex)

    S = np.linalg.inv(np.eye(2, dtype=complex) + Kc @ C_tip)
    K_eff = S @ Kc   # = (K_c⁻¹ + C_tip)⁻¹, symmetric

    EI, H, L = cant.EI, cant.H, cant.L
    mT_over_m = cant.mT / cant.m
    T = (H**2 / EI) * K_eff[0, 0] - (H / 4)**2 * mT_over_m * alpha4 * L
    X = (H / EI) * K_eff[0, 1]
    U = (1.0 / EI) * K_eff[1, 1] - mT_over_m * alpha4 * L

    # Drive forces filtered through the same cone compliance.  The rigid
    # model's F_vert / F_lon are the apex loads along n̂ / t̂; decompose,
    # filter with S, and project back so the existing RHS formulas apply.
    g_b = S @ (F_vert * n_hat + F_lon * t_hat)
    F_vert_eff = n_hat @ g_b
    F_lon_eff = t_hat @ g_b
    return T, X, U, F_vert_eff, F_lon_eff


def distributed_load(x: np.ndarray, cant: CantileverParams,
                     exc: ExcitationParams) -> np.ndarray:
    """
    Distributed electrostatic load q(x) (Eqs. 3–4).

    q(d) = −cos(φ) ε₀ εᵣ V_AC ΔV w(x) / d(x)²
    d(x) = lift_height + H cos(φ) + (L_contact − x) sin(φ)

    The ``lift_height`` term (``exc.lift_height``) is the rigid-body elevation
    of the entire lever above the sample surface.  It is 0.0 in contact mode
    (default), recovering the original Bradler (2016) expression.  In lift or
    non-contact mode set it to the physical tip-sample gap at the tip apex.

    The effective electrostatic width ``w(x)`` equals the body width ``w`` on
    the clamp-to-contact span and ``cant.overhang_width_ratio * w`` on the free
    overhang [contact_x, L].  With the default ratio 1.0 this reduces to the
    original constant-width load.  A ratio below 1 represents a real narrow
    cone/tip region past the contact, which carries far less electrostatic load
    and thereby controls where the displacement electrostatic blind spot
    (D-ESBS) sits relative to the tip (see
    ``CantileverParams.overhang_width_ratio``).

    Parameters
    ----------
    x : array of positions along lever [m], shape (N,)

    Returns
    -------
    q : complex array, shape (N,)

    Raises
    ------
    ValueError
        If d(x) <= 0 at any grid point, which would indicate a physically
        impossible (penetrating) geometry.  The error message reports the
        offending x positions and the lift_height value.
    """
    d_x = (exc.lift_height
           + cant.H * np.cos(cant.phi)
           + (cant.contact_x - x) * np.sin(cant.phi))
    if np.any(d_x <= 0):
        bad_x = x[d_x <= 0]
        raise ValueError(
            f"distributed_load: gap d(x) <= 0 at x = {bad_x} m.  "
            f"lift_height={exc.lift_height:.3e} m may be too small or negative."
        )
    # Effective electrostatic width: full width on the clamp-to-contact span;
    # a real cone/tip region past the contact is narrow along its whole length,
    # so the overhang electrostatic width is reduced by overhang_width_ratio.
    w_eff = np.full_like(np.asarray(x, dtype=float), cant.w)
    r = getattr(cant, "overhang_width_ratio", 1.0)
    if cant.tip_offset > 0.0 and r != 1.0:
        w_eff = np.where(x > cant.contact_x, cant.w * r, cant.w)

    # Negative sign per Eq. 3 / Eq. 4 of Bradler 2016 — electrostatic force
    # acts opposite to the cantilever deflection convention (hence the minus sign).
    q = (-np.cos(cant.phi) * epsilon_0 * exc.eps_r
         * exc.V_AC * exc.delta_V * w_eff / d_x**2)
    return q.astype(complex)


# ---------------------------------------------------------------------------
# BVP formulation
# ---------------------------------------------------------------------------

def _build_ode_rhs(alpha4: complex, q_vals: np.ndarray, x_grid: np.ndarray,
                   EI: float):
    """
    Return a callable for scipy.integrate.solve_bvp representing the 1st-order
    system equivalent to d⁴z/dx⁴ = α⁴ z + q(x)/EI.

    State vector: y = [z, z', z'', z''']  (all complex)

    dy/dx = [z', z'', z''', α⁴ z + q(x)/EI]
    """
    from scipy.interpolate import interp1d
    q_interp = interp1d(x_grid, q_vals, kind='linear', fill_value='extrapolate')

    def ode(x, y):
        # y shape: (4, N_colloc)
        q_x = q_interp(x)
        dy = np.zeros_like(y)
        dy[0] = y[1]
        dy[1] = y[2]
        dy[2] = y[3]
        dy[3] = alpha4 * y[0] + q_x / EI
        return dy

    return ode


def _build_bc(T: complex, X: complex, U: complex,
              F_vert: complex, F_lon: complex,
              cant: CantileverParams, contact: ContactParams) -> callable:
    """
    Return boundary condition residual function for solve_bvp.

    At x=0:  z(0)=0,  z'(0)=0
    At x=L:  z''(L)  = T z'(L) − X z(L) + (H/EI) cosθ F_lon − (H/EI) sinθ F_vert
             z'''(L) = U z(L)  + X z'(L) − (1/EI) cosθ F_vert − (1/EI) sinθ F_lon
    """
    theta = cant.phi + contact.psi
    EI = cant.EI
    H = cant.H
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    rhs_zpp  = (H / EI) * cos_th * F_lon - (H / EI) * sin_th * F_vert
    rhs_zppp = -(1.0 / EI) * cos_th * F_vert - (1.0 / EI) * sin_th * F_lon

    def bc(ya, yb):
        # ya = y at x=0,  yb = y at x=L
        # Bradler 2016 Eqs. 15–16:
        #   z''(L) + T z'(L) + X z(L) = RHS_mom
        #   z'''(L) − U z(L) − X z'(L) = RHS_shear
        res = np.array([
            ya[0],           # z(0) = 0
            ya[1],           # z'(0) = 0
            yb[2] + T * yb[1] + X * yb[0] - rhs_zpp,    # Eq. 15
            yb[3] - U * yb[0] - X * yb[1] - rhs_zppp,   # Eq. 16
        ], dtype=complex)
        return res

    return bc


# ---------------------------------------------------------------------------
# Analytical solver (variation of constants, Supplementary Material)
# ---------------------------------------------------------------------------

def _alpha_from_alpha4(alpha4: complex) -> complex:
    """
    Return the first-quadrant complex 4th root of alpha4.

    We need Re(alpha) > 0 so that e^(alpha*x) is the growing exponential
    and the c-/s- basis is correctly oriented.
    """
    alpha = np.sqrt(np.sqrt(alpha4))
    if np.real(alpha) < 0:
        alpha = -alpha
    return alpha


def _default_obs_x(cant: CantileverParams) -> float:
    """Default observation point: 14 µm back from the contact, clipped safely."""
    x_contact = cant.contact_x
    return x_contact - 14e-6 if x_contact > 14e-6 else 0.5 * x_contact


def _free_basis(alpha: complex, xi) -> tuple:
    """
    Values and first three derivatives of the four homogeneous EB solutions
    [cosh, sinh, cos, sin](α·ξ), stacked as shape-(4, ...) arrays.

    Used for the overhang segment (local coordinate ξ = x − contact_x), where
    all four basis functions are needed because the segment is clamped by
    nothing — its boundary conditions come from continuity/jump at the contact
    and the free end at the physical tip.
    """
    axi = alpha * np.asarray(xi)
    ch, sh = np.cosh(axi), np.sinh(axi)
    co, si = np.cos(axi), np.sin(axi)
    a, a2, a3 = alpha, alpha**2, alpha**3
    z    = np.array([ch,      sh,      co,      si])
    zp   = np.array([a * sh,  a * ch, -a * si,  a * co])
    zpp  = np.array([a2 * ch, a2 * sh, -a2 * co, -a2 * si])
    zppp = np.array([a3 * sh, a3 * ch,  a3 * si, -a3 * co])
    return z, zp, zpp, zppp


def _extract_obs(x: np.ndarray, z: np.ndarray, zp: np.ndarray,
                 obs_x: float) -> tuple:
    """
    Return (A_disp, phase_disp, A_slope, phase_slope) at obs_x [m].

    Always computes both quantities in one pass so the GUI can switch
    between them without recomputing the BVP.
    """
    obs_x = float(np.clip(obs_x, x[0], x[-1]))
    idx = int(np.argmin(np.abs(x - obs_x)))
    zv  = z[idx]
    zpv = zp[idx]
    A_disp  = float(np.abs(zv))
    # Negative sign per Eq. 18 of Bradler 2016.
    ph_disp = float(-np.arctan2(np.imag(zv),  np.real(zv)))
    A_slope  = float(np.abs(zpv))
    ph_slope = float(-np.arctan2(np.imag(zpv), np.real(zpv)))
    return A_disp, ph_disp, A_slope, ph_slope


def _particular_solution(alpha: complex, q_vals: np.ndarray, x: np.ndarray,
                         EI: float) -> tuple:
    """
    Particular solution of z'''' = α⁴ z + q/EI with zero initial conditions
    z = z' = z'' = z''' = 0 at x[0], via the Green's function

        G(x,t) = (sinh(α(x−t)) − sin(α(x−t))) / (2α³)

    evaluated with the 4-exponential cumulative-trapezoid scheme (identical to
    the historical inline implementation in _solve_analytical; extracted so the
    overhang segment can reuse it).  The Green's kernel depends only on x−t, so
    the scheme is translation invariant; coordinates are shifted to x − x[0]
    internally to keep the exponentials well scaled on any segment.

    Returns
    -------
    (y0, y0p, y0pp, y0ppp) : four complex arrays, shape (len(x),)
    """
    n = len(x)
    if np.all(np.abs(q_vals) < 1e-50):
        zero = np.zeros(n, dtype=complex)
        return zero, zero.copy(), zero.copy(), zero.copy()

    from scipy.integrate import cumulative_trapezoid

    a3 = alpha**3
    xl = x - x[0]

    qEI   = q_vals.astype(complex) / EI
    qpEI  = np.gradient(q_vals, x).astype(complex) / EI
    qppEI = np.gradient(qpEI * EI, x).astype(complex) / EI

    e1 = np.exp( alpha * xl); e2 = np.exp( 1j * alpha * xl)
    e3 = np.exp(-alpha * xl); e4 = np.exp(-1j * alpha * xl)

    def _f(e_neg, denom, q):
        return e_neg * q / denom

    def _cumtrap(f):
        return np.concatenate([[0.0 + 0j], cumulative_trapezoid(f, xl)])

    f1 = _f(e3,  4   * a3, qEI)
    f2 = _f(e4, -4j  * a3, qEI)
    f3 = _f(e1, -4   * a3, qEI)
    f4 = _f(e2,  4j  * a3, qEI)

    c1 = _cumtrap(f1); c2 = _cumtrap(f2)
    c3 = _cumtrap(f3); c4 = _cumtrap(f4)

    y0 = c1*e1 + c2*e2 + c3*e3 + c4*e4

    A1 = f1 + alpha    * c1
    A2 = f2 + 1j*alpha * c2
    A3 = f3 - alpha    * c3
    A4 = f4 - 1j*alpha * c4
    y0p = A1*e1 + A2*e2 + A3*e3 + A4*e4

    g1 = _f(e3,  4   * a3, qpEI)
    g2 = _f(e4, -4j  * a3, qpEI)
    g3 = _f(e1, -4   * a3, qpEI)
    g4 = _f(e2,  4j  * a3, qpEI)

    bB1 = g1 + alpha    * A1
    bB2 = g2 + 1j*alpha * A2
    bB3 = g3 - alpha    * A3
    bB4 = g4 - 1j*alpha * A4
    y0pp = bB1*e1 + bB2*e2 + bB3*e3 + bB4*e4

    h1 = _f(e3,  4   * a3, qppEI)
    h2 = _f(e4, -4j  * a3, qppEI)
    h3 = _f(e1, -4   * a3, qppEI)
    h4 = _f(e2,  4j  * a3, qppEI)

    D1 = h1 + alpha    * bB1
    D2 = h2 + 1j*alpha * bB2
    D3 = h3 - alpha    * bB3
    D4 = h4 - 1j*alpha * bB4
    y0ppp = D1*e1 + D2*e2 + D3*e3 + D4*e4

    return y0, y0p, y0pp, y0ppp


def _solve_bc_system_with_diagnostics(M: np.ndarray,
                                      rhs: np.ndarray,
                                      freq: float) -> tuple[np.ndarray, dict]:
    """
    Solve the analytical BC system (2×2 terminal or 6×6 two-segment) and
    report simple conditioning metrics.

    The primary solve remains the existing raw ``np.linalg.solve`` path so we can
    compare any conditioning strategy against the historical behaviour.  In
    parallel, a row/column-equilibrated solve is computed to estimate how much a
    light-touch numerical conditioning step changes the result.

    Parameters
    ----------
    M : (2, 2) complex ndarray
        Boundary-condition matrix.
    rhs : (2,) complex ndarray
        Right-hand side vector.
    freq : float
        Frequency [Hz], used only for diagnostics/error context.

    Returns
    -------
    coeffs : (2,) complex ndarray
        Raw analytical coefficients [Ca, Cb].
    diagnostics : dict
        Conditioning information for downstream inspection.
    """
    diag: dict[str, float | bool | str] = {
        "freq_hz": float(freq),
        "det_abs": float(abs(np.linalg.det(M))),
        "cond_raw": float(np.linalg.cond(M)),
    }

    # Simple equilibration: normalize row/column magnitudes to O(1) without
    # changing the underlying Bradler system.
    row_norms = np.maximum(np.linalg.norm(M, axis=1), 1e-300)
    M_row = M / row_norms[:, None]
    rhs_row = rhs / row_norms

    col_norms = np.maximum(np.linalg.norm(M_row, axis=0), 1e-300)
    M_scaled = M_row / col_norms[None, :]

    diag["cond_row_scaled"] = float(np.linalg.cond(M_row))
    diag["cond_rowcol_scaled"] = float(np.linalg.cond(M_scaled))

    coeffs_scaled = None
    try:
        y_scaled = np.linalg.solve(M_scaled, rhs_row)
        coeffs_scaled = y_scaled / col_norms
        diag["scaled_available"] = True
        diag["residual_scaled"] = float(np.linalg.norm(M @ coeffs_scaled - rhs))
    except np.linalg.LinAlgError:
        diag["scaled_available"] = False
        diag["residual_scaled"] = float("nan")

    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError as exc:
        msg = (
            f"Analytical BC matrix solve failed at f={freq/1e3:.3f} kHz; "
            f"|det(M)|={diag['det_abs']:.3e}, cond_raw={diag['cond_raw']:.3e}, "
            f"cond_row={diag['cond_row_scaled']:.3e}, "
            f"cond_rowcol={diag['cond_rowcol_scaled']:.3e}"
        )
        raise np.linalg.LinAlgError(msg) from exc

    diag["residual_raw"] = float(np.linalg.norm(M @ coeffs - rhs))
    if coeffs_scaled is not None:
        denom = max(float(np.linalg.norm(coeffs)), 1e-300)
        diag["solution_rel_diff_scaled_vs_raw"] = float(
            np.linalg.norm(coeffs_scaled - coeffs) / denom
        )
    else:
        diag["solution_rel_diff_scaled_vs_raw"] = float("nan")

    return coeffs, diag


def _solve_analytical(alpha4: complex,
                      q_vals: np.ndarray,
                      x_grid: np.ndarray,
                      T: complex, X: complex, U: complex,
                      F_vert: complex, F_lon: complex,
                      cant: CantileverParams,
                      contact: ContactParams,
                      freq: float,
                      obs_x: float = None) -> dict:
    """
    Bradler (2016) closed-form EB solution using the za/zb homogeneous basis.

    Homogeneous basis satisfying z(0)=z'(0)=0 (Bradler 2016, below Eq. 14):
        za(x) = (cosh(αx) − cos(αx)) / (2α²)
        zb(x) = (sinh(αx) − sin(αx)) / (2α³)

    Particular solution via Green's function (equivalent to variation of
    constants; cumulative trapezoidal integration on x_grid).

    Boundary conditions at x=L (Bradler 2016, Eqs. 15–16):
        z''(L) + T z'(L) + X z(L) = RHS_mom
        z'''(L) − U z(L) − X z'(L) = RHS_shear

    Always computes both z and z' at obs_x so the GUI can switch between
    displacement and slope display without recomputing.
    """
    alpha = _alpha_from_alpha4(alpha4)
    L   = cant.contact_x
    EI  = cant.EI
    H   = cant.H
    theta  = cant.phi + contact.psi
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    x  = x_grid

    if obs_x is None:
        obs_x = _default_obs_x(cant)

    a2 = alpha**2
    a3 = alpha**3

    # ------------------------------------------------------------------
    # Bradler homogeneous solutions at x = L
    # ------------------------------------------------------------------
    aL   = alpha * L
    sha_L = np.sinh(aL); cha_L = np.cosh(aL)
    sia_L = np.sin(aL);  cia_L = np.cos(aL)

    za_L     = (cha_L - cia_L) / (2*a2)
    za_p_L   = (sha_L + sia_L) / (2*alpha)
    za_pp_L  = (cha_L + cia_L) / 2
    za_ppp_L = alpha * (sha_L - sia_L) / 2

    zb_L     = (sha_L - sia_L) / (2*a3)
    zb_p_L   = (cha_L - cia_L) / (2*a2)    # = za_L
    zb_pp_L  = (sha_L + sia_L) / (2*alpha)  # = za_p_L
    zb_ppp_L = (cha_L + cia_L) / 2          # = za_pp_L

    # ------------------------------------------------------------------
    # Particular solution via Green's function (cumulative trapezoid)
    #
    # The particular solution satisfying z(0)=z'(0)=z''(0)=z'''(0)=0 is
    #   zq(x) = ∫₀ˣ G(x,t) q(t)/EI dt
    # where G(x,t) = (sinh(α(x-t)) − sin(α(x-t))) / (2α³).
    #
    # Expanding the exponentials yields the 4-term cumtrap form below,
    # which is mathematically identical (proven by direct substitution).
    # ------------------------------------------------------------------
    y0, y0p, y0pp, y0ppp = _particular_solution(alpha, q_vals, x, EI)

    # ------------------------------------------------------------------
    # 2×2 BC system (Bradler 2016, Eqs. 15–16):
    #   z''(L) + T z'(L) + X z(L) = RHS_mom
    #   z'''(L) − U z(L) − X z'(L) = RHS_shear
    #
    # z = Ca·za + Cb·zb + zq  →  solve for Ca, Cb
    # ------------------------------------------------------------------
    RHS_mom   = (H / EI) * cos_th * F_lon - (H / EI) * sin_th * F_vert
    RHS_shear = -(1.0 / EI) * cos_th * F_vert - (1.0 / EI) * sin_th * F_lon

    M11 = za_pp_L  + T * za_p_L  + X * za_L
    M12 = zb_pp_L  + T * zb_p_L  + X * zb_L
    M21 = za_ppp_L - U * za_L    - X * za_p_L
    M22 = zb_ppp_L - U * zb_L    - X * zb_p_L

    R1 = RHS_mom   - (y0pp[-1]  + T * y0p[-1]  + X * y0[-1])
    R2 = RHS_shear - (y0ppp[-1] - U * y0[-1]   - X * y0p[-1])

    M = np.array([[M11, M12], [M21, M22]], dtype=complex)
    rhs = np.array([R1, R2], dtype=complex)
    (Ca, Cb), bc_diag = _solve_bc_system_with_diagnostics(M, rhs, freq)

    # ------------------------------------------------------------------
    # Reconstruct full solution on x_grid using za/zb basis
    # ------------------------------------------------------------------
    ax  = alpha * x
    sha = np.sinh(ax); cha = np.cosh(ax)
    sia = np.sin(ax);  cia = np.cos(ax)

    za_x   = (cha - cia) / (2*a2)
    za_p_x = (sha + sia) / (2*alpha)
    zb_x   = (sha - sia) / (2*a3)
    zb_p_x = (cha - cia) / (2*a2)    # = za_x

    # Higher derivatives (needed by BVP path and mode-shape view)
    za_pp_x  = (cha + cia) / 2
    za_ppp_x = alpha * (sha - sia) / 2
    zb_pp_x  = (sha + sia) / (2*alpha)  # = za_p_x
    zb_ppp_x = (cha + cia) / 2           # = za_pp_x

    z    = y0    + Ca * za_x    + Cb * zb_x
    zp   = y0p   + Ca * za_p_x  + Cb * zb_p_x
    zpp  = y0pp  + Ca * za_pp_x + Cb * zb_pp_x
    zppp = y0ppp + Ca * za_ppp_x + Cb * zb_ppp_x

    A_disp, ph_disp, A_slope, ph_slope = _extract_obs(x, z, zp, obs_x)

    return {
        'x': x, 'z': z, 'zp': zp, 'zpp': zpp, 'zppp': zppp,
        'A_disp':    A_disp,    'phase_disp':  ph_disp,
        'A_slope':   A_slope,   'phase_slope': ph_slope,
        # backward-compatible aliases — OBD measures slope, not displacement
        'A': A_slope, 'phase': ph_slope,
        'analytical_bc_diagnostics': bc_diag,
        'freq': freq,
    }


def _solve_analytical_two_segment(alpha4: complex,
                                  q1_vals: np.ndarray, x1_grid: np.ndarray,
                                  q2_vals: np.ndarray, x2_grid: np.ndarray,
                                  T: complex, X: complex, U: complex,
                                  F_vert: complex, F_lon: complex,
                                  cant: CantileverParams,
                                  contact: ContactParams,
                                  freq: float,
                                  obs_x: float = None) -> dict:
    """
    Closed-form two-segment EB solution for a tip set back from the free end
    (cant.tip_offset > 0).

    Geometry (standard Rabe CR-AFM two-region model):
        segment 1: x ∈ [0, x_c], clamped at x = 0
        segment 2: x ∈ [x_c, L], the overhang, with a genuine free end
                   z''(L) = z'''(L) = 0
    The Bradler tip reaction (T, X, U and the excitation right-hand sides)
    acts at the interior point x_c as jump conditions on moment and shear:

        z, z' continuous at x_c
        [z''(x_c⁻) − z''(x_c⁺)]  + T z' + X z  = RHS_mom
        [z'''(x_c⁻) − z'''(x_c⁺)] − U z − X z' = RHS_shear

    In the limit tip_offset → 0 the overhang carries nothing and these reduce
    exactly to the terminal BCs of _solve_analytical (Eqs. 15–16).

    NOTE: this replaces the former _append_overhang cubic extrapolation, which
    propagated the *tip-reaction* curvature and shear into the overhang.  A
    load-free overhang carries only its own inertial moment (~|αL₂|⁴, i.e.
    ≲1 % here), so that construction violated the free-end conditions at the
    physical tip by orders of magnitude and produced spurious displacement
    nodes on the overhang (verified independently against a Hermite-FEM
    discretization of the full beam, July 2026).

    Unknowns: [Ca, Cb] (clamped za/zb basis on segment 1, as in
    _solve_analytical) and [b1..b4] (full cosh/sinh/cos/sin basis on the
    overhang, local coordinate ξ = x − x_c).  Particular solutions with zero
    initial conditions at each segment's left end are added on both segments,
    so q(x) (e.g. the distributed electrostatic load) now also acts on the
    overhang.
    """
    alpha = _alpha_from_alpha4(alpha4)
    Lc  = cant.contact_x
    L2  = cant.tip_offset
    EI  = cant.EI
    H   = cant.H
    theta  = cant.phi + contact.psi
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    if obs_x is None:
        obs_x = _default_obs_x(cant)

    a2 = alpha**2
    a3 = alpha**3

    # --- clamped basis za/zb at the contact point (cf. _solve_analytical) ---
    aL   = alpha * Lc
    sha_L = np.sinh(aL); cha_L = np.cosh(aL)
    sia_L = np.sin(aL);  cia_L = np.cos(aL)

    za_L     = (cha_L - cia_L) / (2*a2)
    za_p_L   = (sha_L + sia_L) / (2*alpha)
    za_pp_L  = (cha_L + cia_L) / 2
    za_ppp_L = alpha * (sha_L - sia_L) / 2

    zb_L     = (sha_L - sia_L) / (2*a3)
    zb_p_L   = (cha_L - cia_L) / (2*a2)
    zb_pp_L  = (sha_L + sia_L) / (2*alpha)
    zb_ppp_L = (cha_L + cia_L) / 2

    # --- particular solutions on both segments (zero ICs at segment start) ---
    y0, y0p, y0pp, y0ppp = _particular_solution(alpha, q1_vals, x1_grid, EI)
    y2, y2p, y2pp, y2ppp = _particular_solution(alpha, q2_vals, x2_grid, EI)

    # --- overhang basis at ξ = 0 and ξ = L2 ---
    phi0, phi0p, phi0pp, phi0ppp = _free_basis(alpha, 0.0)
    phiL, phiLp, phiLpp, phiLppp = _free_basis(alpha, L2)

    # --- excitation right-hand sides (identical to terminal path) ---
    RHS_mom   = (H / EI) * cos_th * F_lon - (H / EI) * sin_th * F_vert
    RHS_shear = -(1.0 / EI) * cos_th * F_vert - (1.0 / EI) * sin_th * F_lon

    # --- 6×6 system for [Ca, Cb, b1, b2, b3, b4] ---
    M   = np.zeros((6, 6), dtype=complex)
    rhs = np.zeros(6, dtype=complex)

    # (1) continuity of z at x_c
    M[0, 0] = za_L;   M[0, 1] = zb_L;   M[0, 2:] = -phi0
    rhs[0] = -y0[-1]
    # (2) continuity of z' at x_c
    M[1, 0] = za_p_L; M[1, 1] = zb_p_L; M[1, 2:] = -phi0p
    rhs[1] = -y0p[-1]
    # (3) moment jump: [z1'' − z2''] + T z' + X z = RHS_mom  (z, z' at x_c)
    M[2, 0] = za_pp_L + T * za_p_L + X * za_L
    M[2, 1] = zb_pp_L + T * zb_p_L + X * zb_L
    M[2, 2:] = -phi0pp
    rhs[2] = RHS_mom - (y0pp[-1] + T * y0p[-1] + X * y0[-1])
    # (4) shear jump: [z1''' − z2'''] − U z − X z' = RHS_shear
    M[3, 0] = za_ppp_L - U * za_L - X * za_p_L
    M[3, 1] = zb_ppp_L - U * zb_L - X * zb_p_L
    M[3, 2:] = -phi0ppp
    rhs[3] = RHS_shear - (y0ppp[-1] - U * y0[-1] - X * y0p[-1])
    # (5)–(6) free end at the physical tip: z''(L) = z'''(L) = 0
    M[4, 2:] = phiLpp
    rhs[4] = -y2pp[-1]
    M[5, 2:] = phiLppp
    rhs[5] = -y2ppp[-1]

    coeffs, bc_diag = _solve_bc_system_with_diagnostics(M, rhs, freq)
    Ca, Cb = coeffs[0], coeffs[1]
    b = coeffs[2:]

    # --- reconstruct segment 1 on x1_grid (za/zb basis) ---
    ax  = alpha * x1_grid
    sha = np.sinh(ax); cha = np.cosh(ax)
    sia = np.sin(ax);  cia = np.cos(ax)

    za_x     = (cha - cia) / (2*a2)
    za_p_x   = (sha + sia) / (2*alpha)
    za_pp_x  = (cha + cia) / 2
    za_ppp_x = alpha * (sha - sia) / 2
    zb_x     = (sha - sia) / (2*a3)
    zb_p_x   = (cha - cia) / (2*a2)
    zb_pp_x  = (sha + sia) / (2*alpha)
    zb_ppp_x = (cha + cia) / 2

    z1    = y0    + Ca * za_x    + Cb * zb_x
    z1p   = y0p   + Ca * za_p_x  + Cb * zb_p_x
    z1pp  = y0pp  + Ca * za_pp_x + Cb * zb_pp_x
    z1ppp = y0ppp + Ca * za_ppp_x + Cb * zb_ppp_x

    # --- reconstruct segment 2 on x2_grid (local ξ, full 4-function basis) ---
    xi = x2_grid - Lc
    P, Pp, Ppp, Pppp = _free_basis(alpha, xi)
    z2    = y2    + b @ P
    z2p   = y2p   + b @ Pp
    z2pp  = y2pp  + b @ Ppp
    z2ppp = y2ppp + b @ Pppp

    # --- concatenate, dropping the duplicated contact point from segment 2 ---
    x    = np.concatenate([x1_grid, x2_grid[1:]])
    z    = np.concatenate([z1,    z2[1:]])
    zp   = np.concatenate([z1p,   z2p[1:]])
    zpp  = np.concatenate([z1pp,  z2pp[1:]])
    zppp = np.concatenate([z1ppp, z2ppp[1:]])

    A_disp, ph_disp, A_slope, ph_slope = _extract_obs(x, z, zp, obs_x)

    return {
        'x': x, 'z': z, 'zp': zp, 'zpp': zpp, 'zppp': zppp,
        'A_disp':    A_disp,    'phase_disp':  ph_disp,
        'A_slope':   A_slope,   'phase_slope': ph_slope,
        # backward-compatible aliases — OBD measures slope, not displacement
        'A': A_slope, 'phase': ph_slope,
        'analytical_bc_diagnostics': bc_diag,
        'freq': freq,
    }


# ---------------------------------------------------------------------------
# Single-frequency solver
# ---------------------------------------------------------------------------

def solve_single_frequency(freq: float,
                            cant: CantileverParams,
                            contact: ContactParams,
                            exc: ExcitationParams,
                            opts: SolverOptions,
                            obs_x: float = None) -> dict:
    """
    Solve the EB BVP at a single frequency.

    Parameters
    ----------
    freq : float
        Frequency [Hz].
    obs_x : float or None
        Observation position [m] from clamped end.  Defaults to L − 14 µm.

    Returns
    -------
    dict with keys:
        'x'          : spatial grid [m], shape (N,)
        'z'          : displacement z(x) [m], complex, shape (N,)
        'zp'         : slope z'(x), complex, shape (N,)
        'zpp'        : curvature z''(x), complex, shape (N,)
        'zppp'       : z'''(x), complex, shape (N,)
        'A_disp'     : |z(obs_x)|  [m]
        'phase_disp' : phase of z(obs_x) [rad]
        'A_slope'    : |z'(obs_x)| [rad]
        'phase_slope': phase of z'(obs_x) [rad]
        'A'          : alias for A_disp (backward compat)
        'phase'      : alias for phase_disp (backward compat)
        'freq'       : frequency [Hz]
    """
    omega = 2.0 * np.pi * freq
    EI = cant.EI
    L = cant.contact_x

    # --- derived quantities ---
    alpha4 = compute_alpha4(omega, cant, opts)
    k1c, k2c = compute_contact_stiffnesses(omega, contact)
    T, X, U = compute_auxiliary(alpha4, k1c, k2c, cant, contact)
    F_vert, F_lon = compute_tip_forces(omega, k1c, k2c, exc)

    # Optional elastic tip cone: replace the rigid-tip contact reaction and
    # drive forces with their compliant-cone counterparts (exactly equal to
    # the above when k_cone_lat = k_cone_ax = inf, the default).
    if (np.isfinite(getattr(contact, 'k_cone_lat', np.inf))
            or np.isfinite(getattr(contact, 'k_cone_ax', np.inf))):
        T, X, U, F_vert, F_lon = compute_effective_contact(
            alpha4, k1c, k2c, cant, contact, F_vert, F_lon)

    # --- spatial grid ---
    x_grid = np.linspace(0.0, L, opts.n_points)

    # --- distributed load ---
    if exc.use_distributed_load:
        q_vals = distributed_load(x_grid, cant, exc)
    else:
        q_vals = np.zeros(opts.n_points, dtype=complex)

    # --- dispatch to chosen solver ---
    if obs_x is None:
        obs_x = _default_obs_x(cant)

    two_segment = cant.tip_offset > 0.0

    if two_segment:
        # Overhang grid [x_c, L]; the distributed electrostatic load also acts
        # on the overhang (the lever remains a conductor above the sample).
        n2 = max(41, int(np.ceil(opts.n_points * cant.tip_offset / L)) + 1)
        x2_grid = np.linspace(L, cant.L, n2)
        if exc.use_distributed_load:
            q2_vals = distributed_load(x2_grid, cant, exc)
        else:
            q2_vals = np.zeros(n2, dtype=complex)

    if opts.method == 'analytical':
        if two_segment:
            return _solve_analytical_two_segment(
                alpha4, q_vals, x_grid, q2_vals, x2_grid,
                T, X, U, F_vert, F_lon, cant, contact, freq, obs_x)
        return _solve_analytical(alpha4, q_vals, x_grid, T, X, U,
                                 F_vert, F_lon, cant, contact, freq, obs_x)

    # --- BVP path: solve using real/imag splitting (solve_bvp is real-valued) ---
    if two_segment:
        x, z, zp, zpp, zppp = _solve_bvp_two_segment(
            alpha4, T, X, U, F_vert, F_lon, cant, contact, exc, opts)
    else:
        ode_fn = _build_ode_rhs(alpha4, q_vals, x_grid, EI)
        bc_fn = _build_bc(T, X, U, F_vert, F_lon, cant, contact)

        result = _solve_bvp_complex(ode_fn, bc_fn, x_grid, opts)

        x = result['x']
        z    = result['y'][0]
        zp   = result['y'][1]
        zpp  = result['y'][2]
        zppp = result['y'][3]

    A_disp, ph_disp, A_slope, ph_slope = _extract_obs(x, z, zp, obs_x)

    return {
        'x': x,
        'z': z,
        'zp': zp,
        'zpp': zpp,
        'zppp': zppp,
        'A_disp':    A_disp,    'phase_disp':  ph_disp,
        'A_slope':   A_slope,   'phase_slope': ph_slope,
        # backward-compatible aliases — OBD measures slope, not displacement
        'A': A_slope, 'phase': ph_slope,
        'freq': freq,
    }


def _solve_bvp_complex(ode_fn: callable, bc_fn: callable,
                       x_grid: np.ndarray, opts: SolverOptions) -> dict:
    """
    Wrap scipy solve_bvp to handle complex-valued ODEs by splitting into
    real and imaginary parts (8-component real system).
    """
    n = 4  # state dimension

    def ode_real(x, y):
        # y: shape (8, N) — [Re(y0..y3), Im(y0..y3)]
        y_c = y[:n] + 1j * y[n:]
        dy_c = ode_fn(x, y_c)
        return np.vstack([np.real(dy_c), np.imag(dy_c)])

    def bc_real(ya, yb):
        ya_c = ya[:n] + 1j * ya[n:]
        yb_c = yb[:n] + 1j * yb[n:]
        res_c = bc_fn(ya_c, yb_c)
        return np.concatenate([np.real(res_c), np.imag(res_c)])

    # Initial guess: linear profile (small)
    x0 = x_grid
    y0 = np.zeros((2 * n, len(x0)))

    sol = solve_bvp(ode_real, bc_real, x0, y0, tol=1e-6, max_nodes=5000,
                    verbose=0)

    if not sol.success:
        # Retry with tighter initial guess shaped by wavenumber
        y0_retry = np.zeros((2 * n, len(x0)))
        y0_retry[0] = (x0 / x0[-1])**2 * 1e-12
        sol = solve_bvp(ode_real, bc_real, x0, y0_retry, tol=1e-5,
                        max_nodes=3000, verbose=0)

    x_sol = sol.x
    y_c = sol.y[:n] + 1j * sol.y[n:]

    return {'x': x_sol, 'y': y_c, 'success': sol.success, 'message': sol.message}


def _solve_bvp_two_segment(alpha4: complex,
                           T: complex, X: complex, U: complex,
                           F_vert: complex, F_lon: complex,
                           cant: CantileverParams,
                           contact: ContactParams,
                           exc: ExcitationParams,
                           opts: SolverOptions) -> tuple:
    """
    scipy.solve_bvp two-segment counterpart of _solve_analytical_two_segment,
    providing a genuinely independent numerical cross-check of the interior
    jump conditions and free-end treatment.

    Both segments are mapped onto t ∈ [0, 1] and solved simultaneously as a
    16-component real system (4 complex states per segment):
        segment 1: x = t·x_c        (clamped at t=0)
        segment 2: x = x_c + t·L₂   (free end at t=1)
    Interface conditions couple segment 1 at t=1 to segment 2 at t=0.

    Returns
    -------
    (x, z, zp, zpp, zppp) on the concatenated physical grid [0, L].
    """
    xc = cant.contact_x
    L2 = cant.tip_offset
    EI = cant.EI
    H  = cant.H
    theta  = cant.phi + contact.psi
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    RHS_mom   = (H / EI) * cos_th * F_lon - (H / EI) * sin_th * F_vert
    RHS_shear = -(1.0 / EI) * cos_th * F_vert - (1.0 / EI) * sin_th * F_lon

    if exc.use_distributed_load:
        def q_of_x(xv):
            return distributed_load(np.atleast_1d(xv), cant, exc)
    else:
        def q_of_x(xv):
            return np.zeros(np.atleast_1d(xv).shape, dtype=complex)

    def ode(t, Y):
        # Y: (16, N) real = [Re y1(4), Im y1(4), Re y2(4), Im y2(4)]
        y1 = Y[0:4] + 1j * Y[4:8]
        y2 = Y[8:12] + 1j * Y[12:16]
        q1 = q_of_x(t * xc)
        q2 = q_of_x(xc + t * L2)
        d1 = np.vstack([y1[1], y1[2], y1[3],
                        alpha4 * y1[0] + q1 / EI]) * xc
        d2 = np.vstack([y2[1], y2[2], y2[3],
                        alpha4 * y2[0] + q2 / EI]) * L2
        return np.vstack([np.real(d1), np.imag(d1),
                          np.real(d2), np.imag(d2)])

    def bc(Ya, Yb):
        y1a = Ya[0:4] + 1j * Ya[4:8]     # segment 1 at x = 0
        y2a = Ya[8:12] + 1j * Ya[12:16]  # segment 2 at x = x_c⁺
        y1b = Yb[0:4] + 1j * Yb[4:8]     # segment 1 at x = x_c⁻
        y2b = Yb[8:12] + 1j * Yb[12:16]  # segment 2 at x = L
        res = np.array([
            y1a[0],                    # clamp: z(0) = 0
            y1a[1],                    # clamp: z'(0) = 0
            y1b[0] - y2a[0],           # continuity of z at x_c
            y1b[1] - y2a[1],           # continuity of z' at x_c
            (y1b[2] - y2a[2]) + T * y1b[1] + X * y1b[0] - RHS_mom,
            (y1b[3] - y2a[3]) - U * y1b[0] - X * y1b[1] - RHS_shear,
            y2b[2],                    # free end: z''(L) = 0
            y2b[3],                    # free end: z'''(L) = 0
        ], dtype=complex)
        return np.concatenate([np.real(res), np.imag(res)])

    t0 = np.linspace(0.0, 1.0, max(120, opts.n_points // 2))
    Y0 = np.zeros((16, len(t0)))
    sol = solve_bvp(ode, bc, t0, Y0, tol=1e-6, max_nodes=20000, verbose=0)

    # Dense output on the requested resolution
    n1 = opts.n_points
    n2 = max(41, int(np.ceil(opts.n_points * L2 / xc)) + 1)
    t1 = np.linspace(0.0, 1.0, n1)
    t2 = np.linspace(0.0, 1.0, n2)
    Y1 = sol.sol(t1)
    Y2 = sol.sol(t2)
    y1 = Y1[0:4] + 1j * Y1[4:8]
    y2 = Y2[8:12] + 1j * Y2[12:16]

    x    = np.concatenate([t1 * xc, (xc + t2 * L2)[1:]])
    z    = np.concatenate([y1[0], y2[0][1:]])
    zp   = np.concatenate([y1[1], y2[1][1:]])
    zpp  = np.concatenate([y1[2], y2[2][1:]])
    zppp = np.concatenate([y1[3], y2[3][1:]])
    return x, z, zp, zpp, zppp


# ---------------------------------------------------------------------------
# Eigenfrequency utilities (κ-map support)
# ---------------------------------------------------------------------------

def det_bc_matrix(omega: float,
                  cant: CantileverParams,
                  contact: ContactParams,
                  opts: SolverOptions) -> float:
    """
    Absolute value of the 2×2 boundary-condition matrix determinant.

    The 2×2 matrix M arises from imposing the tip boundary conditions
    (Eqs. 15–16) on the two homogeneous basis functions that satisfy
    z(0)=z'(0)=0. Its determinant vanishes exactly at contact-resonance
    eigenfrequencies, regardless of excitation.

    Pass this to a scalar minimiser over frequency to locate modes without
    a full BVP solve — useful for the κ-map sweep.

    NOTE: this uses the *terminal* (contact-at-free-end) formulation.  For
    cant.tip_offset > 0 it neglects the free overhang beyond the contact,
    which shifts eigenfrequencies only marginally (the overhang carries
    ~|αL₂|⁴ relative moment) but is not exact; the forced-response solver
    (solve_single_frequency) uses the full two-segment model.  It also uses
    the rigid-tip T, X, U — a finite contact.k_cone_lat/k_cone_ax (elastic
    tip cone) is honored only by solve_single_frequency.

    Parameters
    ----------
    omega : float
        Angular frequency [rad/s].
    cant, contact, opts : parameter dataclasses.

    Returns
    -------
    float : |det(M)|
    """
    alpha4 = compute_alpha4(omega, cant, opts)
    k1c, k2c = compute_contact_stiffnesses(omega, contact)
    T, X, U = compute_auxiliary(alpha4, k1c, k2c, cant, contact)
    alpha = _alpha_from_alpha4(alpha4)
    aL = alpha * cant.contact_x

    cos_aL = np.cos(aL);  cosh_aL = np.cosh(aL)
    sin_aL = np.sin(aL);  sinh_aL = np.sinh(aL)

    cm_L  = cos_aL  - cosh_aL
    sm_L  = sin_aL  - sinh_aL
    dcm_L = -sin_aL - sinh_aL
    dsm_L =  cos_aL - cosh_aL
    d2cm_L = -cos_aL - cosh_aL
    d2sm_L = -sin_aL - sinh_aL
    d3cm_L =  sin_aL - sinh_aL
    d3sm_L = -cos_aL - cosh_aL

    a = alpha
    # Row 1 imposes Eq. 15 (z'' + T z' + X z = 0): the T term enters with a
    # PLUS sign (a minus sign would shift the eigenfrequencies whenever the
    # H²·k/EI rotational stiffness term is non-negligible).
    mat = np.array([
        [a**2*d2cm_L + T*(a*dcm_L) + X*cm_L,  a**2*d2sm_L + T*(a*dsm_L) + X*sm_L],
        [a**3*d3cm_L - U*cm_L - X*(a*dcm_L),  a**3*d3sm_L - U*sm_L - X*(a*dsm_L)],
    ], dtype=complex)

    return float(abs(np.linalg.det(mat)))


def mode_shape_at_tip(omega_eig: float,
                      cant: CantileverParams,
                      contact: ContactParams,
                      opts: SolverOptions) -> tuple:
    """
    Return (z_L, zp_L) — tip displacement and slope of the contact-resonance
    mode shape at the given (undamped) eigenfrequency.

    Computed from the null vector of the 2×2 BC matrix (smallest singular
    value of M) via SVD.

    Normalisation convention: the mode shape is scaled so that the
    displacement at the beam midpoint |z(L/2)| = 1.  This makes z(L)
    and z'(L) physically comparable across different κ values without the
    trivial result that arises from normalising by z(L) itself.  At the
    zero-crossing (κ ≈ 28.4) z'(L) → 0 while z(L) stays finite, so
    the midpoint provides a stable, non-vanishing reference.

    Fallback: if z(L/2) < 1e-12 (Mode 2 node near midpoint) normalise by
    |z(L)|; if that is also small, normalise by |z'(L)|.

    Parameters
    ----------
    omega_eig : float
        (Undamped) eigenfrequency [rad/s], typically found via det_bc_matrix.
    cant, contact, opts : parameter dataclasses.
        contact should have c1=c2=0 for a real-valued mode shape.

    Returns
    -------
    z_L  : complex   z(L)  of the mode shape, normalised by |z(L/2)|
    zp_L : complex   z'(L) of the mode shape, normalised by |z(L/2)|
    """
    alpha4 = compute_alpha4(omega_eig, cant, opts)
    k1c, k2c = compute_contact_stiffnesses(omega_eig, contact)
    T, X, U = compute_auxiliary(alpha4, k1c, k2c, cant, contact)
    alpha = _alpha_from_alpha4(alpha4)
    aL = alpha * cant.contact_x

    cos_aL = np.cos(aL);  cosh_aL = np.cosh(aL)
    sin_aL = np.sin(aL);  sinh_aL = np.sinh(aL)

    cm_L  = cos_aL  - cosh_aL
    sm_L  = sin_aL  - sinh_aL
    dcm_L = -sin_aL - sinh_aL
    dsm_L =  cos_aL - cosh_aL
    d2cm_L = -cos_aL - cosh_aL
    d2sm_L = -sin_aL - sinh_aL
    d3cm_L =  sin_aL - sinh_aL
    d3sm_L = -cos_aL - cosh_aL

    # Interior basis functions at L/4, L/2, 3L/4 for robust normalization
    def _zx(frac):
        ax = alpha * cant.contact_x * frac
        return (np.cos(ax) - np.cosh(ax)) * B2_ref + (np.sin(ax) - np.sinh(ax)) * B4_ref

    a = alpha
    # T term with PLUS sign per Eq. 15.
    mat = np.array([
        [a**2*d2cm_L + T*(a*dcm_L) + X*cm_L,  a**2*d2sm_L + T*(a*dsm_L) + X*sm_L],
        [a**3*d3cm_L - U*cm_L - X*(a*dcm_L),  a**3*d3sm_L - U*sm_L - X*(a*dsm_L)],
    ], dtype=complex)

    # Null vector via SVD: last row of V^H → [B2, B4] coefficients
    _, _, Vh = np.linalg.svd(mat)
    B2, B4 = Vh[-1]
    B2_ref, B4_ref = B2, B4   # alias for _zx closure

    z_L  = B2 * cm_L   + B4 * sm_L
    zp_L = B2 * (a * dcm_L) + B4 * (a * dsm_L)

    # Robust interior norm: max |z| at L/4, L/2, 3L/4
    # (avoids nodes of any single mode, e.g. z(L/2)→0 for Mode 2 at κ≈249)
    norm = max(abs(_zx(0.25)), abs(_zx(0.50)), abs(_zx(0.75)))
    _TOL = 1e-12
    if norm < _TOL:
        norm = abs(z_L)
    if norm < _TOL:
        norm = abs(zp_L)
    if norm > _TOL:
        z_L  = z_L  / norm
        zp_L = zp_L / norm

    return complex(z_L), complex(zp_L)


# ---------------------------------------------------------------------------
# Frequency sweep
# ---------------------------------------------------------------------------

def frequency_sweep(cant: CantileverParams,
                    contact: ContactParams,
                    exc: ExcitationParams,
                    opts: SolverOptions,
                    obs_x: float = None,
                    verbose: bool = False) -> dict:
    """
    Compute amplitude and phase spectra over a frequency range.

    Always computes both displacement |z| and slope |z'| at obs_x so
    the GUI can switch between them without recomputing the BVP.

    Returns
    -------
    dict with keys:
        'freqs'            : frequency array [Hz], shape (Nf,)
        'amplitudes_disp'  : |z(obs_x)| for each frequency  [m],  shape (Nf,)
        'phases_disp'      : phase of z(obs_x) [deg], shape (Nf,)
        'amplitudes_slope' : |z'(obs_x)| for each frequency [rad], shape (Nf,)
        'phases_slope'     : phase of z'(obs_x) [deg], shape (Nf,)
        'amplitudes'       : alias for amplitudes_slope (backward compat)
        'phases'           : alias for phases_slope     (backward compat)
        'results'          : list of per-frequency dicts (spatial profiles)
    """
    freqs = opts.frequencies
    Nf = len(freqs)
    amps_disp   = np.zeros(Nf)
    phs_disp    = np.zeros(Nf)
    amps_slope  = np.zeros(Nf)
    phs_slope   = np.zeros(Nf)
    results = []

    for i, f in enumerate(freqs):
        if verbose and i % 20 == 0:
            print(f"  f = {f/1e3:.1f} kHz  ({i+1}/{Nf})")
        res = solve_single_frequency(f, cant, contact, exc, opts, obs_x)
        amps_disp[i]  = res['A_disp']
        phs_disp[i]   = np.degrees(res['phase_disp'])
        amps_slope[i] = res['A_slope']
        phs_slope[i]  = np.degrees(res['phase_slope'])
        results.append(res)

    return {
        'freqs':            freqs,
        'amplitudes_disp':  amps_disp,
        'phases_disp':      phs_disp,
        'amplitudes_slope': amps_slope,
        'phases_slope':     phs_slope,
        'amplitudes':       amps_slope,  # backward compat — OBD = slope
        'phases':           phs_slope,   # backward compat — OBD = slope
        'results':          results,
    }


def analytical_conditioning_sweep(cant: CantileverParams,
                                  contact: ContactParams,
                                  exc: ExcitationParams,
                                  opts: SolverOptions,
                                  obs_x: float = None,
                                  verbose: bool = False) -> dict:
    """
    Sweep the analytical solver and collect BC-system conditioning metrics.

    This utility is intended for diagnosing where the closed-form Bradler solve
    becomes numerically fragile. It does not modify the analytical path or hide
    failures. If a frequency point cannot be solved analytically, the exception
    is recorded and the sweep continues.

    Parameters
    ----------
    cant, contact, exc, opts : parameter dataclasses
    obs_x : float or None
        Observation position [m]. Defaults to the solver's standard value.
    verbose : bool
        If True, print sparse progress updates.

    Returns
    -------
    dict with keys:
        'freqs' : frequency array [Hz]
        'cond_raw' : raw 2x2 BC-matrix condition number
        'cond_row_scaled' : row-scaled condition number
        'cond_rowcol_scaled' : row+column-scaled condition number
        'det_abs' : |det(M)| for the analytical BC matrix
        'residual_raw' : raw BC solve residual norm
        'residual_scaled' : scaled BC solve residual norm (nan if unavailable)
        'solution_rel_diff_scaled_vs_raw' : relative coefficient difference
        'success' : boolean mask, True where the analytical solve succeeded
        'errors' : list of error strings (empty string on success)
        'results' : per-frequency solver result dicts, or None on failure
    """
    if opts.method != 'analytical':
        raise ValueError("analytical_conditioning_sweep requires opts.method='analytical'")

    freqs = opts.frequencies
    Nf = len(freqs)

    cond_raw = np.full(Nf, np.nan, dtype=float)
    cond_row = np.full(Nf, np.nan, dtype=float)
    cond_rowcol = np.full(Nf, np.nan, dtype=float)
    det_abs = np.full(Nf, np.nan, dtype=float)
    residual_raw = np.full(Nf, np.nan, dtype=float)
    residual_scaled = np.full(Nf, np.nan, dtype=float)
    rel_diff = np.full(Nf, np.nan, dtype=float)
    success = np.zeros(Nf, dtype=bool)
    errors = [""] * Nf
    results = [None] * Nf

    for i, f in enumerate(freqs):
        if verbose and i % 20 == 0:
            print(f"  cond sweep: f = {f/1e3:.1f} kHz  ({i+1}/{Nf})")
        try:
            res = solve_single_frequency(f, cant, contact, exc, opts, obs_x)
            diag = res.get('analytical_bc_diagnostics', {})
            cond_raw[i] = diag.get('cond_raw', np.nan)
            cond_row[i] = diag.get('cond_row_scaled', np.nan)
            cond_rowcol[i] = diag.get('cond_rowcol_scaled', np.nan)
            det_abs[i] = diag.get('det_abs', np.nan)
            residual_raw[i] = diag.get('residual_raw', np.nan)
            residual_scaled[i] = diag.get('residual_scaled', np.nan)
            rel_diff[i] = diag.get('solution_rel_diff_scaled_vs_raw', np.nan)
            success[i] = True
            results[i] = res
        except Exception as exc_info:
            errors[i] = str(exc_info)

    return {
        'freqs': freqs,
        'cond_raw': cond_raw,
        'cond_row_scaled': cond_row,
        'cond_rowcol_scaled': cond_rowcol,
        'det_abs': det_abs,
        'residual_raw': residual_raw,
        'residual_scaled': residual_scaled,
        'solution_rel_diff_scaled_vs_raw': rel_diff,
        'success': success,
        'errors': errors,
        'results': results,
    }


# ---------------------------------------------------------------------------
# SHO peak fitting
# ---------------------------------------------------------------------------

def sho_analysis(sweep_result: dict,
                 prominence_frac: float = 0.01,
                 use_phase: bool = True,
                 quantity: str = 'slope') -> list:
    """
    Find all resonance peaks in a frequency sweep and fit each to an SHO model.

    The SHO amplitude model is::

        A(f) = A_res / sqrt( (Q*(1-(f/f0)^2))^2 + (f/f0)^2 )

    which gives A = A_res exactly at f = f0, with Q the quality factor.

    Parameters
    ----------
    sweep_result : dict
        Output of frequency_sweep().
    prominence_frac : float
        Minimum peak prominence as a fraction of the global max amplitude.
        Peaks below this threshold are ignored.  Default 0.01 (1 %).
        Lower values catch weak peaks at high contact stiffness; increase to
        suppress noise peaks in noisy spectra.
    use_phase : bool
        If True (default), supplement amplitude-based peak detection with
        phase-slope detection.  At resonance the unwrapped phase drops
        steeply regardless of drive amplitude, making this robust at high k₁
        where the amplitude peak can fall below ``prominence_frac``.
        Phase-only candidates are kept only if the amplitude has a local
        maximum next to them, so anti-resonances (amplitude minima, which
        also carry a steep phase change) are not reported as modes.
    quantity : {'slope', 'displacement'}
        Which observable to analyse: ``'slope'`` (default) uses |z'(obs_x)|,
        the optical-beam-deflection (OBD) signal; ``'displacement'`` uses
        |z(obs_x)|.

    Returns
    -------
    list of dict, sorted by ascending frequency (mode 1 = lowest).  Each dict:
        'f_res'     : fitted resonance frequency [Hz]
        'A_res'     : peak amplitude at f_res (from SHO fit)
        'Q'         : quality factor (nan if fit did not converge)
        'phase_res' : phase of the analysed observable at f_res [deg]
        'converged' : True if scipy.optimize.curve_fit succeeded
    """
    import warnings
    from scipy.signal import find_peaks, peak_widths
    from scipy.optimize import curve_fit

    q = str(quantity).lower()
    if q in ('slope', 's', 'obd'):
        amp_key, ph_key = 'amplitudes_slope', 'phases_slope'
    elif q in ('displacement', 'disp', 'd', 'z'):
        amp_key, ph_key = 'amplitudes_disp', 'phases_disp'
    else:
        raise ValueError("quantity must be 'slope' or 'displacement'")
    freqs  = sweep_result['freqs']
    amps   = sweep_result.get(amp_key, sweep_result.get('amplitudes'))
    phases = sweep_result.get(ph_key, sweep_result.get('phases'))
    df     = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    N      = len(freqs)

    def _sho_amp(f, f0, Q, A_res):
        x = f / f0
        return A_res / np.sqrt((Q * (1.0 - x * x)) ** 2 + x * x)

    # ---- 1. Amplitude-based peak detection ----
    # Require peaks to be separated by at least 1 % of the sweep span so that
    # sub-peaks on the same broad feature are not double-counted.
    min_dist  = max(3, N // 100)
    min_prom  = prominence_frac * np.max(amps)
    amp_peaks, _ = find_peaks(amps, prominence=min_prom, distance=min_dist)

    # ---- 2. Phase-slope-based peak detection (supplemental) ----
    # At resonance the unwrapped phase falls steeply; |d(phase)/df| peaks there.
    # This catches resonances whose amplitude is low but phase signature is clear.
    phase_peaks = np.array([], dtype=int)
    if use_phase and N > 10:
        phase_unwrap = np.unwrap(np.radians(phases))
        # Negative gradient = falling phase = approaching resonance
        slope = np.gradient(phase_unwrap, freqs)          # rad / Hz
        neg_slope = -slope                                 # positive at resonance
        # Prominence threshold: 10 % of the maximum phase-slope magnitude
        ph_prom = 0.10 * float(np.max(neg_slope)) if np.max(neg_slope) > 0 else 0.0
        if ph_prom > 0:
            phase_peaks, _ = find_peaks(neg_slope,
                                        prominence=ph_prom,
                                        distance=min_dist)
        # Keep only phase candidates that sit on an amplitude maximum.  An
        # anti-resonance also produces a steep phase change, but there the
        # amplitude has a minimum; snap genuine candidates onto the local
        # amplitude maximum and drop the rest.
        snapped = []
        for idx in phase_peaks:
            lo = max(int(idx) - min_dist, 0)
            hi = min(int(idx) + min_dist + 1, N)
            j = lo + int(np.argmax(amps[lo:hi]))
            if lo < j < hi - 1:
                snapped.append(j)
        phase_peaks = np.array(sorted(set(snapped)), dtype=int)

    # ---- 3. Merge candidate indices, deduplicate within min_dist ----
    all_candidates = np.unique(np.concatenate([amp_peaks, phase_peaks]))
    # Remove candidates that are within min_dist of a stronger amplitude peak
    kept = []
    for idx in all_candidates:
        too_close = any(
            abs(idx - k) < min_dist and amps[k] > amps[idx]
            for k in kept
        )
        if not too_close:
            kept.append(int(idx))
    peak_idx = np.array(sorted(kept), dtype=int)

    # Fall back to global maximum if nothing survived
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(amps))])

    modes = []
    for ip in peak_idx:
        f_pk = freqs[ip]
        A_pk = amps[ip]

        # Estimate half-power bandwidth → initial Q guess.
        # Clamp bw to at least 2 grid steps so Q_init stays finite.
        with warnings.catch_warnings():
            # Very sharp or edge peaks trigger harmless width warnings.
            warnings.simplefilter("ignore")
            w_pts, _, _, _ = peak_widths(amps, [ip], rel_height=0.5)
        bw_hz  = max(float(w_pts[0]) * df, 2.0 * df)
        # Maximum Q reliably resolvable with this frequency step:
        # need ~5 points across the FWHM = f_pk/Q, so Q_max = f_pk/(5*df).
        Q_max  = min(f_pk / (5.0 * df), 50000.0)
        Q_init = float(np.clip(f_pk / bw_hz, 2.0, Q_max))

        # Fitting window: ±3× estimated half-bandwidth, at least ±8 steps
        half_win = max(3.0 * bw_hz, 8.0 * df)
        mask = (freqs >= f_pk - half_win) & (freqs <= f_pk + half_win)
        if mask.sum() < 7:
            lo = max(ip - 8, 0)
            hi = min(ip + 9, len(freqs))
            mask = np.zeros(len(freqs), dtype=bool)
            mask[lo:hi] = True

        f_win = freqs[mask]
        A_win = amps[mask]

        try:
            # Fit in normalised units (f / f_pk, A / A_pk).  Amplitudes are
            # ~1e-9 m or rad, so fitting raw values lets the optimiser stop
            # at the initial guess because the residuals are already tiny.
            popt, _ = curve_fit(
                _sho_amp, f_win / f_pk, A_win / A_pk,
                p0=[1.0, Q_init, 1.0],
                bounds=([f_win[0] / f_pk, 1.0, 0.0],
                        [f_win[-1] / f_pk, Q_max, np.inf]),
                maxfev=10000,
            )
            f0_fit = popt[0] * f_pk
            Q_fit = popt[1]
            A_fit = popt[2] * A_pk
            # If Q hit the resolution ceiling the result is unreliable;
            # fall back to bandwidth estimate and flag as not converged.
            if Q_fit >= 0.99 * Q_max:
                f0_fit, Q_fit, A_fit = f_pk, Q_init, A_pk
                converged = False
            else:
                converged = True
        except Exception:
            # Fall back to bandwidth-based estimate rather than returning nan
            f0_fit, Q_fit, A_fit = f_pk, Q_init, A_pk
            converged = False

        phase_at = float(np.interp(f0_fit, freqs, phases))
        modes.append({
            'f_res':     f0_fit,
            'A_res':     A_fit,
            'Q':         Q_fit,
            'phase_res': phase_at,
            'converged': converged,
        })

    # Sort by frequency ascending so mode index 0 is always the lowest
    # contact-resonance mode, not the loudest.  This keeps labelling consistent
    # through anti-crossings where modes swap amplitude dominance.
    modes.sort(key=lambda m: m['f_res'])
    return modes
