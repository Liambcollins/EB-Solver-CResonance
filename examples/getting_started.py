"""
Getting started with eb_cr_afm
==============================

A five-minute tour of the solver.  Run it from the repository root:

    python examples/getting_started.py

It prints a few numbers and saves three figures to ``results/``:

1. ``getting_started_1_transfer_function.png``  free vs in-contact spectra
2. ``getting_started_2_mode_shape.png``         the lever shape at resonance
3. ``getting_started_3_null_spots.png``         null spot vs electrostatic
                                                 blind spot near the tip

Everything is in SI units (metres, newtons, hertz, kilograms).
"""
import dataclasses
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")            # write PNG files; no window needed
import matplotlib.pyplot as plt

# Make the package importable even if it has not been pip-installed.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from eb_cr_afm import (
    ContactParams, ExcitationParams, SolverOptions,
    default_params_multi75, default_params_null_spot_paper,
    frequency_sweep, sho_analysis, solve_single_frequency,
)

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def strongest_mode(sweep, quantity="slope"):
    """Return the fitted resonance with the largest amplitude."""
    modes = sho_analysis(sweep, quantity=quantity)
    return max(modes, key=lambda m: m["A_res"])


# ---------------------------------------------------------------------------
# 1. A cantilever, a contact, a drive and solver settings
# ---------------------------------------------------------------------------
# Every simulation needs four parameter objects.  The presets fill them with
# sensible values for a BudgetSensors Multi75-G probe (k = 2.8 N/m, f0 = 75 kHz).
cant, contact, exc, opts = default_params_multi75()
print(f"Lever: L = {cant.L*1e6:.0f} um, k = {cant.k} N/m, "
      f"tilt = {np.degrees(cant.phi):.0f} deg")

# Free cantilever: switch the tip-sample springs off (k1 = k2 = 0).  With no
# contact the sample drive d1 cannot reach the lever, so drive it instead with
# a small force f0 = 1 nN at the tip.
free = dataclasses.replace(contact, k1=0.0, k2=0.0)
tip_force = ExcitationParams(d1=0.0, d2=0.0, f0=1e-9, use_f0=True,
                             V_AC=0.0, delta_V=0.0, use_distributed_load=False)
opts_free = dataclasses.replace(opts, f_start=60e3, f_stop=90e3, f_step=10.0)
sweep_free = frequency_sweep(cant, free, tip_force, opts_free)
m_free = strongest_mode(sweep_free)
print(f"Free resonance:     {m_free['f_res']/1e3:6.1f} kHz  (Q = {m_free['Q']:.0f})")

# In contact with a k1 = 100 N/m sample (the preset), drive the sample
# vertically by d1 = 1 pm, as in PFM.
sweep_cr = frequency_sweep(cant, contact, exc, opts)
m_cr = strongest_mode(sweep_cr)
# The peak is only a few hundred Hz wide.  Zoom in with a 10 Hz step so the
# SHO fit returns a reliable Q (a coarse step makes Q look too small).
opts_cr = dataclasses.replace(opts, f_start=m_cr["f_res"] - 5e3,
                              f_stop=m_cr["f_res"] + 5e3, f_step=10.0)
m_cr = strongest_mode(frequency_sweep(cant, contact, exc, opts_cr))
print(f"Contact resonance:  {m_cr['f_res']/1e3:6.1f} kHz  (Q = {m_cr['Q']:.0f}), "
      f"k1 = {contact.k1:.0f} N/m")

# The two sweeps use different drives (a tip force vs a sample displacement),
# so each curve is normalised to its own maximum.
fig, ax = plt.subplots(figsize=(6, 3.5))
for sw, lab in [(sweep_free, "free (tip-force drive)"),
                (sweep_cr, f"in contact, k1 = {contact.k1:.0f} N/m (d1 drive)")]:
    a = sw["amplitudes_slope"]
    ax.semilogy(sw["freqs"]/1e3, a / a.max(), label=lab)
ax.set_xlabel("Frequency (kHz)")
ax.set_ylabel("Tip slope |z'| (normalised)")
ax.set_title("Transfer function (what an optical-lever AFM measures)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "getting_started_1_transfer_function.png"), dpi=150)

# ---------------------------------------------------------------------------
# 2. The shape of the lever at the contact resonance
# ---------------------------------------------------------------------------
res = solve_single_frequency(m_cr["f_res"], cant, contact, exc, opts_cr)
x_um = res["x"] * 1e6            # position along the lever, clamp = 0
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
a1.plot(x_um, np.abs(res["z"]) * 1e12)
a1.set_ylabel("|z(x)| (pm)")
a1.set_title(f"Mode shape at {m_cr['f_res']/1e3:.1f} kHz")
a2.plot(x_um, np.abs(res["zp"]), color="C1")
a2.set_ylabel("|z'(x)| (rad)")
a2.set_xlabel("Position along lever x (um)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "getting_started_2_mode_shape.png"), dpi=150)

# ---------------------------------------------------------------------------
# 3. Null spot (D-NS) and electrostatic blind spot (D-ESBS)
# ---------------------------------------------------------------------------
# The paper preset sets the tip 10.9 um back from the free end, so the lever
# has a short free "overhang" beyond the tip.  Two drives act on it:
#   * piezo drive  - the sample moves by d1 (the signal you want), and
#   * electrostatic drive - the AC bias pulls on the whole lever (an artefact).
# D-NS:   where the piezo-driven displacement vanishes at the contact resonance.
# D-ESBS: where the electrostatically driven displacement vanishes (measured
#         well below resonance, here 30 kHz).
cant_p, contact_p, _, opts_p = default_params_null_spot_paper()
contact_p = ContactParams(k1=1500.0, k2=0.0, c1=20e-6, c2=0.0)  # frictionless contact
piezo = ExcitationParams(d1=3e-12, d2=0.0, f0=0.0, use_f0=False,
                         V_AC=0.0, delta_V=0.0, use_distributed_load=False)
electro = ExcitationParams(d1=0.0, d2=0.0, f0=0.0, use_f0=False,
                           V_AC=3.0, delta_V=1.0, use_distributed_load=True)

sweep_p = frequency_sweep(cant_p, contact_p, piezo,
                          dataclasses.replace(opts_p, f_start=300e3, f_stop=600e3,
                                              f_step=250.0))
f_cr = strongest_mode(sweep_p, quantity="displacement")["f_res"]
fine = SolverOptions(n_points=1500, g=opts_p.g)
r_pz = solve_single_frequency(f_cr, cant_p, contact_p, piezo, fine)
r_es = solve_single_frequency(30e3, cant_p, contact_p, electro, fine)


def near_tip_null(r, cant, window=70e-6):
    """Deepest displacement minimum within `window` of the free end.

    Returns (position, depth); depth is |z| at the minimum relative to the
    maximum |z| on the lever, so a value near 0 means a true null.
    """
    from scipy.interpolate import CubicSpline
    x = r["x"]
    xs = np.linspace(cant.L - window, cant.L, 20001)
    z = CubicSpline(x, r["z"].real)(xs) + 1j * CubicSpline(x, r["z"].imag)(xs)
    i = int(np.argmin(np.abs(z)))
    return xs[i], np.abs(z[i]) / np.abs(r["z"]).max()


x_ns, depth_ns = near_tip_null(r_pz, cant_p)
x_esbs, depth_esbs = near_tip_null(r_es, cant_p)
d_tip = lambda x: (cant_p.contact_x - x) * 1e6   # um behind the tip (+) / past it (-)
print(f"Contact resonance (paper preset, k1 = {contact_p.k1:.0f} N/m): {f_cr/1e3:.1f} kHz")
print(f"D-NS   is {d_tip(x_ns):+.2f} um from the tip (depth {depth_ns:.1e})")
print(f"D-ESBS is {d_tip(x_esbs):+.2f} um from the tip (depth {depth_esbs:.1e})")
print("        (+ = toward the clamp, - = on the overhang past the tip)")

fig, ax = plt.subplots(figsize=(6, 3.5))
xu = r_pz["x"] * 1e6
ax.semilogy(xu, np.abs(r_pz["z"]) / np.abs(r_pz["z"]).max(),
            label=f"piezo, on resonance ({f_cr/1e3:.0f} kHz)")
ax.semilogy(xu, np.abs(r_es["z"]) / np.abs(r_es["z"]).max(),
            label="electrostatic, 30 kHz")
ax.axvline(cant_p.contact_x * 1e6, color="k", ls="--", lw=0.8, label="tip position")
ax.set_xlim(cant_p.L * 1e6 - 60, cant_p.L * 1e6)
ax.set_ylim(1e-3, 2)
ax.set_xlabel("Position along lever x (um)")
ax.set_ylabel("|z(x)| (normalised)")
ax.set_title("Near-tip displacement nulls")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "getting_started_3_null_spots.png"), dpi=150)

print(f"\nFigures written to {os.path.abspath(OUT)}")
