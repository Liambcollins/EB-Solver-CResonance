"""
Parameter dataclasses for the Bradler et al. (2016) EB contact-resonance AFM model.

All quantities are in SI base units unless otherwise noted in the field docstring.
References
----------
Bradler et al., J. Appl. Phys. 120, 165107 (2016)  [Paper 1]
Bradler et al., J. Appl. Phys. 122, 065106 (2017)  [Paper 2]
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CantileverParams:
    """
    Geometric and mechanical properties of the AFM cantilever.

    All lengths in metres, masses in kg, spring constant in N/m.

    Parameters
    ----------
    L : float
        Cantilever length [m].  Table I: 240e-6 m.
    w : float
        Cantilever width [m].  Table I: 40e-6 m.
    H : float
        Tip height [m].  Table I: 6e-6 m.
    k : float
        Static spring constant [N/m].  Table I: 3 N/m.
        Used to derive EI = k * L**3 / 3.
    m : float
        Total lever mass [kg].  Table I: 5e-11 kg.
    mT : float
        Tip mass [kg].  Table I: 5e-12 kg.
        Positioned at H/4 from the lever plane (Rabe convention, Eq. 8).
    phi : float
        Lever tilt angle [rad].  Table I: 0.2 rad.
    tip_offset : float
        Axial offset of the tip/contact point from the beam free end [m].
        ``tip_offset = 0`` reproduces the original ``x = L`` contact model.
        Positive values shift the contact point toward the clamp so that
        ``contact_x = L - tip_offset``.
    overhang_width_ratio : float
        Effective electrostatic width of the free overhang [contact_x, L] as a
        fraction of the body width ``w``.  The width used in the distributed
        electrostatic load is ``w`` on the clamp-to-contact span and
        ``overhang_width_ratio * w`` on the overhang.

        The uniform-beam model (``overhang_width_ratio = 1.0``, default) treats
        the overhang as full width, so it carries the full-width electrostatic
        load — which over-loads the overhang relative to a real tapered tip and
        pushes the displacement electrostatic blind spot (D-ESBS) back toward
        the contact.  Real tips taper to little area past the contact; setting
        ``overhang_width_ratio`` well below 1 (from SEM area or an FEM match)
        moves the D-ESBS out toward the tip, reproducing the ordering seen in
        realistic finite-element models.  It affects ONLY the distributed
        electrostatic load, not the beam mechanics (mass/stiffness), consistent
        with the tip material contributing negligible flexural rigidity.
    """
    L: float = 240e-6      # m
    w: float = 40e-6       # m
    H: float = 6e-6        # m
    k: float = 3.0         # N/m
    m: float = 5e-11       # kg
    mT: float = 5e-12      # kg
    phi: float = 0.2       # rad
    tip_offset: float = 0.0  # m
    overhang_width_ratio: float = 1.0  # dimensionless (1 = full-width overhang)

    @property
    def EI(self) -> float:
        """Flexural rigidity derived from static spring constant [N·m²].
        EI = k * L³ / 3  (Eq. after Eq. 2, Paper 1)
        """
        return self.k * self.L**3 / 3.0

    @property
    def contact_x(self) -> float:
        """Contact position measured from the clamp [m]."""
        x_contact = self.L - self.tip_offset
        if not (0.0 < x_contact <= self.L):
            raise ValueError(
                "Cantilever tip_offset must satisfy 0 <= tip_offset < L so the "
                "contact point remains within the beam."
            )
        return x_contact

    @property
    def L1(self) -> float:
        """Paper-style alias for the clamp-to-contact distance [m]."""
        return self.contact_x

    @property
    def L2(self) -> float:
        """Paper-style alias for the contact-to-free-end overhang [m]."""
        return self.tip_offset


@dataclass
class ContactParams:
    """
    Contact mechanics parameters (tip–sample interface).

    Parameters
    ----------
    k1 : float
        Vertical contact stiffness [N/m].  Table I: 1000 N/m.
    k2 : float
        Longitudinal contact stiffness [N/m].  Table I: 850 N/m.
    c1 : float
        Vertical contact damping coefficient [N·s/m].  Table I: 20e-6 N·s/m.
    c2 : float
        Longitudinal contact damping coefficient [N·s/m].  Table I: 17e-6 N·s/m.
    psi : float
        Local sample tilt angle [rad].  Table I: 0 rad.
        Contact angle theta = phi + psi  (Eq. 7, Paper 1).
    k_cone_lat : float
        Lateral (beam-axial direction) bending stiffness of the elastic tip
        cone at its apex [N/m].  Default ``inf`` reproduces the original
        Bradler rigid-tip model exactly.  A finite value places the cone
        compliance in series with the contact springs via a 2x2 apex force
        balance (see solver.compute_effective_contact), which both (i)
        saturates the rotational constraint the lateral contact spring can
        impose on the beam and (ii) releases the rigid-tip kinematic locking
        z(x_c) = H tan(theta) z'(x_c) under a stiff normal contact, allowing
        drive-dependent near-tip null positions.  Order 3*E*I_cone/H^3, a few
        kN/m for a ~12 um Si cone.
    k_cone_ax : float
        Axial (along the cone, beam-normal direction) stiffness of the tip
        cone at its apex [N/m].  Default ``inf`` (rigid); the axial stiffness
        of a solid cone is orders of magnitude above the lateral bending
        stiffness, so ``inf`` is usually appropriate.
    """
    k1: float = 1000.0     # N/m
    k2: float = 850.0      # N/m
    c1: float = 20e-6      # N·s/m  (20 μN/(m/s))
    c2: float = 17e-6      # N·s/m  (17 μN/(m/s))
    psi: float = 0.0       # rad
    k_cone_lat: float = np.inf  # N/m (inf = rigid tip, original model)
    k_cone_ax: float = np.inf   # N/m (inf = rigid tip, original model)


@dataclass
class ExcitationParams:
    """
    Excitation sources acting on the cantilever/tip.

    Parameters
    ----------
    d1 : float
        Vertical sample displacement amplitude [m].  Table I: 1e-12 m (1 pm).
    d2 : float
        Longitudinal sample displacement amplitude [m].  Table I: 0.
    f0 : complex
        Local tip-force amplitude [N]. May be complex to encode a phase.
    delta_V : float
        DC potential difference between lever and sample [V].  Table I: 1 V.
        Sum of applied DC voltage and surface potential.
    V_AC : float
        Applied AC bias amplitude [V].  Table I: 1 V.
    eps_r : float
        Relative permittivity of the gap medium (air ≈ 1).  Table I: 1.
    use_distributed_load : bool
        If True, include the distributed electrostatic load q(x).
    use_f0 : bool
        If True, include the local tip electrostatic force f0.
    bias_coupled_f0 : bool
        If True, add a local electrostatic force
        ``0.5 * capacitance_gradient * V_bias**2`` to the tip forcing, where
        ``V_bias = delta_V + V_AC``.
    capacitance_gradient : float
        Constant capacitance gradient ``dC/dz`` [F/m] used for the local
        square-law electrostatic tip force. Default: 50 nF/m.
    lift_height : float
        Rigid-body offset of the entire lever above the sample surface [m].
        Used in the distributed-load gap formula:
            d(x) = lift_height + H cos(φ) + (L_contact − x) sin(φ)
        Set to 0.0 (default) for contact mode, where the tip apex is at
        the surface and this formula reduces to the original Bradler (2016)
        expression.  For lift-mode or non-contact operation set this to the
        physical tip-sample gap at the tip end of the lever.
    """
    d1: float = 1e-12      # m  (1 pm)
    d2: float = 0.0        # m
    f0: complex = 100e-12  # N  (100 pN)
    delta_V: float = 1.0   # V
    V_AC: float = 1.0      # V
    eps_r: float = 1.0
    use_distributed_load: bool = True
    use_f0: bool = True
    bias_coupled_f0: bool = False
    capacitance_gradient: float = 50e-9
    lift_height: float = 0.0  # m


@dataclass
class SolverOptions:
    """
    Numerical solver configuration.

    Parameters
    ----------
    n_points : int
        Number of spatial grid points along the cantilever (x ∈ [0, L]).
        Increase for better accuracy with stiff contacts or many modes.
    f_start : float
        Start frequency of the sweep [Hz].  Table I: 300e3 Hz.
    f_stop : float
        Stop frequency of the sweep [Hz].  Table I: 400e3 Hz.
    f_step : float
        Frequency step [Hz].  Table I: 500 Hz.
    g : float
        Air damping coefficient eta [1/s].  Table I lists eta in units of
        1000/s, value 20  →  20 000 1/s.
        Enters alpha^4 = (m / (EI*L)) * (omega^2 - i*g*omega).
    method : str
        Solver backend.  'analytical' (default) uses the closed-form
        variation-of-constants solution (fast, no iteration); 'bvp' uses
        scipy.integrate.solve_bvp as an independent numerical check.
    """
    n_points: int = 500
    f_start: float = 300e3   # Hz
    f_stop: float = 400e3    # Hz
    f_step: float = 500.0    # Hz
    g: float = 20000.0       # 1/s  (Table I: 20 x 1000/s)
    method: str = 'analytical'  # 'analytical' (default, fast) or 'bvp'

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency array for the sweep [Hz]."""
        return np.arange(self.f_start, self.f_stop + self.f_step / 2, self.f_step)
