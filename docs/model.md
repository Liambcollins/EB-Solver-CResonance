# The two-segment Euler–Bernoulli model

This note describes the physics implemented in `eb_cr_afm`. It extends the
contact-resonance model of Bradler *et al.* (J. Appl. Phys. **120**, 165107
(2016); **122**, 065106 (2017)) to a tilted cantilever whose tip is set back
from the free end. The model and its consequences for null spots and
electrostatic blind spots are presented in

> M. Checa, H. Swicegood, R. Millan-Solsona, R. Bulanadi, J. Zhao, J. Lasseter,
> S. Jesse and L. Collins, "Dynamics of Null and Electrostatic Blind Spots for
> Quantitative PFM", *ACS Nano* (2026). [arXiv:2607.27486](https://arxiv.org/abs/2607.27486)

**Abbreviations.** D-NS / S-NS: displacement / slope null spot (the point on
the lever where the resonant, piezo-driven response vanishes). D-ESBS /
S-ESBS: displacement / slope electrostatic blind spot (the point where the
electrostatically driven response vanishes). OBD: optical beam deflection,
which measures the slope z′. IDS: interferometric displacement sensing,
which measures the displacement z.

## Model formulation

Cantilever dynamics were modelled with an extension of the analytical Euler–Bernoulli (EB) contact-resonance framework of Bradler *et al.* [Bradler2016; Bradler2017]. The base model treats a clamped, tilted beam (tilt φ, length L, flexural rigidity EI, mass per unit length m/L, air damping g) carrying a rigid tip of height H and lumped mass m_T, coupled to the sample through normal and lateral contact stiffnesses and dashpots (k₁, k₂, c₁, c₂), and driven by any combination of localized excitation at the tip–sample junction (sample displacements d₁, d₂; local force f₀) and a distributed electrostatic load

> q(x) = −cos φ · ε₀ ε_r V_AC ΔV w / d(x)²,  d(x) = H cos φ + (x_c − x) sin φ,  (1)

where d(x) is the local lever–sample gap and x_c the position of the tip along the beam. In Bradler's original formulation the tip coincides with the free end (x_c = L), and the tip–sample interaction enters through terminal boundary conditions containing the auxiliary functions T, X, U [Bradler2016, Eqs. (8)–(16)]:

> z″(L) + T z′(L) + X z(L) = R_M,  z‴(L) − U z(L) − X z′(L) = R_F,  (2)

with R_M, R_F the moment- and force-like excitation terms.

Real probes, however, have the tip set back from the free end by Δx ≈ 10–20 µm (typically confirmed by SEM), leaving a free *overhang* of length L₂ = Δx beyond the contact. Because Eqs. (2) are terminal conditions, they are valid only where the beam ends; for an interior contact the correct model is a **two-segment beam** (the standard two-region geometry of contact-resonance AFM [Rabe1996; Rabe2006]): segment 1 on [0, x_c], clamped at the base, and segment 2 on [x_c, L] carrying no tip load and terminating in a genuine free end. Both segments obey the same EB equation z⁗ = α⁴z + q/EI, with α⁴ = (m/EIL)(ω² − igω). The tip reaction enters as interior jump conditions at x_c:

> z, z′ continuous at x_c,
> [z″(x_c⁻) − z″(x_c⁺)] + T z′(x_c) + X z(x_c) = R_M,  (3)
> [z‴(x_c⁻) − z‴(x_c⁺)] − U z(x_c) − X z′(x_c) = R_F,
> z″(L) = z‴(L) = 0.  (4)

Equations (3)–(4) reduce exactly to Bradler's terminal conditions (2) as Δx → 0, since a zero-length free overhang carries neither moment nor shear. The distributed load (1) acts on **both** segments; note that on the overhang the gap d(x) *decreases* toward the free end (the lever is tilted toward the surface past the contact), so the per-length electrostatic load is largest there — its omission is not a small error. The system is solved in closed form: the clamped basis pair of Bradler on segment 1, the full {cosh, sinh, cos, sin}(αξ) basis on the overhang (local coordinate ξ = x − x_c), Green's-function particular solutions on both segments, and a 6×6 linear system for the six coefficients.

**Validation.** (i) For Δx → 0 the solution converges to the Bradler terminal solution (relative field difference ~10⁻⁴ at Δx = 1 nm). (ii) An independent Hermite finite-element discretization of the full beam — consistent mass, weak-form tip coupling K_tip = EI·[[U, X],[X, T]] applied at the contact node, free end as the natural boundary condition — reproduces the terminal solver to ~10⁻⁵ at Δx = 0 and the two-segment null positions to ≤3×10⁻⁴ in x/L across all contact stiffnesses tested. (iii) A numerically independent shooting/collocation path (scipy `solve_bvp` on the mapped two-segment domain) agrees with the closed form to <1% in the fields, with and without distributed load.

**A caution on naive overhang treatments.** It is tempting to solve the beam on [0, x_c] with the terminal conditions (2) and then extrapolate the deflection across the overhang (e.g., by a cubic Taylor series in the contact-point derivatives). This is incorrect: it assigns the *tip-reaction* moment and shear to the overhang, whereas a load-free overhang can carry only its own inertial moment, a fraction ~|αL₂|⁴ ≈ 10⁻² of the tip reaction. The extrapolated field then violates the free-end conditions z″(L) = z‴(L) = 0 by many orders of magnitude and manufactures spurious displacement nodes on precisely the stretch of beam where the near-tip nulls live — including a false second node and fold (branch annihilation) in D-NS versus k₁ sweeps at k₂ ∝ k₁, and a false convergence of the D-NS to the contact point at high load. The solver in this repository always uses the two-segment model when `tip_offset > 0`.

## Why the overhang and tilt are essential for near-tip nulls

The displacement null spot (D-NS) and displacement electrostatic blind spot (D-ESBS) both lie within a few micrometres of the tip — i.e., **on the overhang**. Three pieces of physics, each absent from the terminal (tip-at-end, untilted) model, control their positions:

1. **Tilt sets the frictionless high-stiffness limit.** When only the surface-normal direction is pinned (k₂ → 0, or an effective lateral stiffness that saturates due to tip-shank compliance), the hard-contact constraint is cos θ·z = H sin θ·z′ at x_c: the rigid tip pivots about its pinned apex. The near-tip displacement zero then sits at a fixed offset **H tan φ behind the contact point** (≈ 2–3 µm for H = 10–15 µm and φ ≈ 11°). Both D-NS and D-ESBS converge to this point, not to the tip.

2. **The overhang carries the nulls, and its physics is the ratio z/z′.** Because the free overhang is nearly straight (curvature suppressed by |αL₂|⁴), the local field is z(x_c) + z′(x_c)·ξ and every near-tip null position is set by the *ratio* of residual displacement to residual slope at the contact. With strongly coupled lateral stiffness (k₂ = βk₁, rigid tip), both z and z′ vanish together as 1/k₁ and the D-NS saturates at a finite offset **past** the contact, ξ* = −z/z′, with the closed form ξ* = −βH²F/M at zero tilt (F, M the modal reaction force and moment). The D-NS then *never* reaches the contact point, while the quasistatic D-ESBS does — a **persistent D-NS/D-ESBS separation** (~3 µm for β = 1 and Multi75-type geometries) that survives the hard-indentation limit. The EB-versus-FEM difference in convergence behaviour is therefore controlled primarily by the lateral contact stiffness, with realistic geometry refining the numbers; conversely, the measured high-load D-NS offset from the contact point is a direct experimental probe of the effective lateral-coupling regime.

3. **The electrostatic load on the overhang can eliminate the blind spot.** The overhang is the part of the lever closest to the sample, so it takes the largest electrostatic load per unit length (for H = 6 µm the end gap is ≈ 2.9 µm and the local load ~4× that above the contact). At soft-to-moderate contact (k₁ ≲ 3 kN/m for Multi75-type geometries) this loading removes the interior D-ESBS altogether: the electrostatically driven displacement has no true zero on the lever (residual minima of 9–19%). A blind-spot "parking" protocol is then ill-defined at low load, which rationalizes both the strong load- and friction-sensitivity of experimental D-ESBS determination and the practical necessity of the hard-indentation regime for quantitative work.

Together these effects mean that quantitative statements about near-tip nulls — their positions, their separation, and their convergence or non-convergence — *require* the tilted two-segment model: the terminal model puts both nulls at the tip by construction, and naive overhang extrapolation puts them at artifact positions.

---

## Where these effects live in the code

| Physics | Parameter | Notes |
|---|---|---|
| Tip set back from the free end | `CantileverParams.tip_offset` | `0` = original Bradler model; `> 0` switches to the two-segment solver |
| Lever tilt | `CantileverParams.phi` | radians |
| Tapered overhang (electrostatics only) | `CantileverParams.overhang_width_ratio` | `1` = full-width overhang; smaller values move the D-ESBS toward the tip |
| Lateral contact coupling | `ContactParams.k2` | `0` = frictionless; `k1` = rigid isotropic contact |
| Elastic tip cone | `ContactParams.k_cone_lat`, `k_cone_ax` | `inf` = rigid tip (default) |
| Electrostatic load on the lever | `ExcitationParams.use_distributed_load`, `V_AC`, `delta_V` | acts on both segments |

The unit tests in `tests/test_two_segment.py` and `tests/test_elastic_tip.py`
check the Δx → 0 limit, agreement with the `bvp` backend, and the free-end
conditions.

## References

- **[Bradler2016]** S. Bradler, S. R. Kachel, A. Schirmeisen and B. Roling, "A theoretical model for the cantilever motion in contact-resonance atomic force microscopy and its application to phase calibration in piezoresponse force and electrochemical strain microscopy", *J. Appl. Phys.* **120**, 165107 (2016). [doi:10.1063/1.4964942](https://doi.org/10.1063/1.4964942)
- **[Bradler2017]** S. Bradler, A. Schirmeisen and B. Roling, "Amplitude quantification in contact-resonance-based voltage-modulated force spectroscopy", *J. Appl. Phys.* **122**, 065106 (2017). [doi:10.1063/1.4998435](https://doi.org/10.1063/1.4998435)
- **[Rabe1996]** U. Rabe, K. Janser and W. Arnold, "Vibrations of free and surface-coupled atomic force microscope cantilevers: Theory and experiment", *Rev. Sci. Instrum.* **67**, 3281 (1996).
- **[Rabe2006]** U. Rabe, "Atomic Force Acoustic Microscopy", in *Applied Scanning Probe Methods II*, Springer (2006).
