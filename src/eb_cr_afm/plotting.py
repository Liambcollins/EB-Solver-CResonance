"""
Visualization utilities for EB contact-resonance AFM results.

All plots are designed to be publication-ready with clear axis labels
and physical units.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional


def plot_transfer_function(sweep_result: dict,
                           title: str = "Transfer Function",
                           figsize: tuple = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot amplitude and phase vs frequency (tip transfer function).

    Parameters
    ----------
    sweep_result : dict
        Output of frequency_sweep().
    """
    freqs = sweep_result['freqs'] / 1e3  # kHz
    amps  = sweep_result['amplitudes']
    phases = sweep_result['phases']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(freqs, amps * 1e12, 'b-', linewidth=1.5)
    ax1.set_ylabel("Amplitude |z'(L)| (pm/rad)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, phases, 'r-', linewidth=1.5)
    ax2.set_ylabel("Phase ∠z'(L) (°)")
    ax2.set_xlabel("Frequency (kHz)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_displacement_profile(res: dict,
                               title: str = "Cantilever Profile",
                               figsize: tuple = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot z(x), z'(x), z''(x), z'''(x) for a single frequency result.
    """
    x_um = res['x'] * 1e6  # µm
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    labels = ["z(x)  (pm)", "z'(x)  (arb)", "z''(x)  (arb)", "z'''(x)  (arb)"]
    keys   = ['z', 'zp', 'zpp', 'zppp']
    scales = [1e12, 1, 1, 1]

    for ax, key, lbl, sc in zip(axes.flat, keys, labels, scales):
        y = res[key] * sc
        ax.plot(x_um, np.real(y), 'b-', label='Re', linewidth=1.5)
        ax.plot(x_um, np.imag(y), 'r--', label='Im', linewidth=1.5)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel(lbl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title}  —  f = {res['freq']/1e3:.1f} kHz", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_2d_spectra(spectra: dict,
                    figsize: tuple = (14, 10),
                    save_path=None):
    """
    2D maps: x=position (um), y=frequency (kHz, log scale).
    Amplitude panels use plasma (log color), phase panels use coolwarm.
    """
    import matplotlib.colors as mcolors
    import numpy as np

    freqs_kHz = spectra["freqs"] / 1e3
    x_um = spectra["x_grid"] * 1e6

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)

    datasets = [
        ("z_amp",    "z(x,f) Amplitude (m)",   True,  "plasma"),
        ("z_phase",  "z(x,f) Phase (deg)",      False, "coolwarm"),
        ("zp_amp",   "z'(x,f) Amplitude (m)",  True,  "plasma"),
        ("zp_phase", "z'(x,f) Phase (deg)",    False, "coolwarm"),
    ]

    for idx, (key, title, is_amp, cmap) in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        data = spectra[key]  # shape (Nx, Nf)

        if is_amp:
            data_plot = np.clip(np.abs(data), 1e-17, None)
            norm = mcolors.LogNorm(vmin=data_plot.max()*1e-6, vmax=data_plot.max())
            im = ax.pcolormesh(x_um, freqs_kHz, data_plot.T,
                               norm=norm, cmap=cmap, shading="auto")
            plt.colorbar(im, ax=ax, label="m")
        else:
            im = ax.pcolormesh(x_um, freqs_kHz, data.T,
                               vmin=-180, vmax=180, cmap=cmap, shading="auto")
            plt.colorbar(im, ax=ax, label="deg")

        ax.set_yscale("log")
        ax.set_xlabel("Position x (um)", fontsize=11)
        ax.set_ylabel("Frequency (kHz)", fontsize=11)
        ax.set_title(title, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_kappa_map(km: dict,
                   cant=None,
                   figsize: tuple = (9, 11),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Three-panel κ = k₁ L³ / EI contact-resonance map.

    (a) Resonance frequency vs κ — shows how each mode shifts from the
        free-cantilever limit (κ=0) to the clamped-pinned limit (κ→∞).
    (b) Normalised OBD slope |z'(L)| vs κ (log y-axis) — the sharp dip
        to zero at κ ≈ 28.4 (Mode 1) and κ ≈ 249 (Mode 2) marks the
        "DART blind spot": the mode exists but the photodetector sees nothing.
    (c) Normalised tip displacement |z(L)| vs κ — remains large and nonzero
        at both zero-crossings, confirming the cantilever is still vibrating.

    Grey dashed vertical lines and blue/orange shading mark the zero-crossing
    positions.  A green dotted line marks κ = 3 (when k₁ = k_c).

    Parameters
    ----------
    km : dict
        Output of kappa_map().
    cant : CantileverParams, optional
        Used only to annotate the κ = 3 reference (k₁ = k_c line).
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.ticker as ticker

    kappa    = km['kappa']
    f_res    = km['f_res']
    zp_norm  = km['zp_norm']
    zL_norm  = km['zL_norm']
    f_free   = km['f_free']
    f_pinned = km['f_pinned']
    n_modes  = len(f_res)

    KC_M1, KC_M2 = 28.44, 248.98        # analytical zero-crossing positions
    COLORS = ['C0', 'C1', 'C2']
    VC     = dict(color='0.45', lw=1.2, ls='--', alpha=0.65)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        r'Contact-resonance map:  $\kappa = k_1 L^3\!/EI$',
        fontsize=13, fontweight='bold', y=0.985)

    # ── (a) Resonance frequency ───────────────────────────────────────────────
    ax = axes[0]
    for i in range(n_modes):
        ax.plot(kappa, np.array(f_res[i]) / 1e3,
                color=COLORS[i], lw=2.2, label=f'Mode {i+1}')
        ax.axhline(f_free[i]   / 1e3, color=COLORS[i], ls=':',  lw=1.1, alpha=0.6)
        ax.axhline(f_pinned[i] / 1e3, color=COLORS[i], ls='-.', lw=1.1, alpha=0.6)

    # Limit-line legend entries (one each)
    ax.plot([], [], color='0.5', ls=':',  lw=1.1, alpha=0.7, label='free-air limit')
    ax.plot([], [], color='0.5', ls='-.', lw=1.1, alpha=0.7, label='pinned limit (κ→∞)')

    ax.axvline(KC_M1, **VC)
    if n_modes >= 2:
        ax.axvline(KC_M2, **VC)
    ax.set_ylabel('Resonance frequency (kHz)', fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.set_ylim(0, None)
    ax.grid(True, which='major', alpha=0.3)
    ax.set_title('(a)', loc='left', fontweight='bold')

    ylim_a = ax.get_ylim()
    ax.text(KC_M1 + 1, ylim_a[1] * 0.06,
            r'$\kappa\!\approx\!28.4$', fontsize=8, color='0.4', ha='left')
    if n_modes >= 2:
        ax.text(KC_M2 + 2, ylim_a[1] * 0.06,
                r'$\kappa\!\approx\!249$', fontsize=8, color='0.4', ha='left')

    # ── (b) OBD slope |z'(L)| — log, normalised ──────────────────────────────
    ax = axes[1]
    for i in range(n_modes):
        ax.semilogy(kappa, zp_norm[i], color=COLORS[i], lw=2.2,
                    label=rf"Mode {i+1}  $|z'(L)|$")
    ax.axvline(KC_M1, **VC)
    if n_modes >= 2:
        ax.axvline(KC_M2, **VC)
    ax.set_ylabel(r'OBD slope  $|z^\prime(L)|$  (norm.)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='major', alpha=0.3)
    ax.set_title('(b)', loc='left', fontweight='bold')

    # Annotations for zero-crossings
    y0_m1 = float(zp_norm[0][np.argmin(np.abs(kappa - KC_M1))])
    ax.annotate(r"$z'(L)\!\to\!0$" + '\n' + r'$\lambda\!\to\!\pi$',
                xy=(KC_M1, max(y0_m1, 1e-3)),
                xytext=(KC_M1 + 15, 5e-2),
                fontsize=8, color=COLORS[0], ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS[0], lw=1.1))
    if n_modes >= 2:
        y0_m2 = float(zp_norm[1][np.argmin(np.abs(kappa - KC_M2))])
        ax.annotate(r"$z'(L)\!\to\!0$" + '\n' + r'$\lambda\!\to\!2\pi$',
                    xy=(KC_M2, max(y0_m2, 1e-3)),
                    xytext=(KC_M2 + 15, 5e-2),
                    fontsize=8, color=COLORS[1], ha='left',
                    arrowprops=dict(arrowstyle='->', color=COLORS[1], lw=1.1))

    # ── (c) Tip displacement |z(L)| — log, normalised ────────────────────────
    ax = axes[2]
    for i in range(n_modes):
        ax.semilogy(kappa, zL_norm[i], color=COLORS[i], lw=2.2,
                    label=rf'Mode {i+1}  $|z(L)|$')
    ax.axvline(KC_M1, **VC)
    if n_modes >= 2:
        ax.axvline(KC_M2, **VC)
    ax.set_xlabel(r'Dimensionless contact stiffness  $\kappa = k_1 L^3\!/EI$',
                  fontsize=11)
    ax.set_ylabel(r'Tip displacement  $|z(L)|$  / interior ref  (norm.)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, which='major', alpha=0.3)
    ax.set_title('(c)', loc='left', fontweight='bold')
    ax.set_xlim(float(kappa[0]), float(kappa[-1]))

    # Shared shading at zero-crossing regions
    for ax in axes:
        ax.axvspan(KC_M1 - 2,  KC_M1 + 2,  alpha=0.08, color=COLORS[0])
        if n_modes >= 2:
            ax.axvspan(KC_M2 - 6,  KC_M2 + 6,  alpha=0.08, color=COLORS[1])

    # Optional κ = 3  (k₁ = k_c)  reference line
    if cant is not None:
        # κ = k1*L³/EI; when k1=cant.k → κ = cant.k*L³/(cant.k*L³/3) = 3
        for ax in axes:
            ax.axvline(3.0, color='green', lw=0.9, ls=':', alpha=0.55)
        axes[0].text(3.3, axes[0].get_ylim()[1] * 0.85,
                     r'$k_1 = k_c$', fontsize=8, color='green',
                     ha='left', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_stiffness_sweep(sweep_result: dict,
                          figsize: tuple = (10, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot resonance frequency and quality factor vs contact stiffness k1.
    """
    k1 = sweep_result['k1']
    f_res = sweep_result['f_res'] / 1e3
    Q = sweep_result['Q']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.semilogx(k1, f_res, 'b-o', markersize=4)
    ax1.set_xlabel("k₁ (N/m)")
    ax1.set_ylabel("f_res (kHz)")
    ax1.set_title("Contact Resonance vs Stiffness")
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(k1, Q, 'r-o', markersize=4)
    ax2.set_xlabel("k₁ (N/m)")
    ax2.set_ylabel("Q factor")
    ax2.set_title("Q Factor vs Contact Stiffness")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_stiffness_frequency_map(sf_map, cant=None, position_um=None,
                                  figsize=(14, 10), cmap_amp="coolwarm",
                                  cmap_phase="plasma", amp_vmin=None, amp_vmax=None,
                                  save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    freqs_kHz = sf_map["freqs"] / 1e3
    k1 = sf_map["k1"]
    amp = sf_map["amp_map"]
    phase = sf_map["phase_map"]
    pos_str = (" at position %d um" % int(position_um)) if position_um is not None else ""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    amp_plot = np.clip(amp, 1e-17, None)
    vmin_amp = amp_vmin if amp_vmin is not None else amp_plot.max() * 1e-6
    vmax_amp = amp_vmax if amp_vmax is not None else amp_plot.max()
    norm_amp = mcolors.LogNorm(vmin=vmin_amp, vmax=vmax_amp)
    im0 = ax.pcolormesh(freqs_kHz, k1, amp_plot, norm=norm_amp, cmap=cmap_amp, shading="auto")
    ax.set_xlim(freqs_kHz.min(), freqs_kHz.max())
    ax.set_ylim(k1.min(), k1.max())
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (kHz)", fontsize=12)
    ax.set_ylabel("k1 (N/m)", fontsize=12)
    ax.set_title("2D Spectra" + pos_str + "\nAmplitude (m)", fontsize=12)
    cb0 = plt.colorbar(im0, ax=ax)
    cb0.set_label("Displacement Amplitude (m)", fontsize=10)
    if cant is not None:
        k = cant.k
        fmid = freqs_kHz[len(freqs_kHz) // 5]
        for label, ytext, yline in [("Free", 0.05*k, 0.1*k), ("Soft Ind.", k, 10*k), ("Hard Ind.", 200*k, 1000*k)]:
            if k1.min() < yline < k1.max():
                ax.axhline(yline, color="white", lw=0.8, ls="--", alpha=0.5)
            if k1.min() < ytext < k1.max():
                ax.text(fmid, ytext, label, color="white", fontsize=9, va="center", alpha=0.9)
    ax2 = axes[1]
    im1 = ax2.pcolormesh(freqs_kHz, k1, phase, vmin=-180, vmax=180, cmap=cmap_phase, shading="auto")
    ax2.set_xlim(freqs_kHz.min(), freqs_kHz.max())
    ax2.set_ylim(k1.min(), k1.max())
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (kHz)", fontsize=12)
    ax2.set_ylabel("k1 (N/m)", fontsize=12)
    ax2.set_title("2D Spectra" + pos_str + "\nPhase (deg)", fontsize=12)
    plt.colorbar(im1, ax=ax2).set_label("Phase (deg)", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig



def plot_modes_and_spectra(cant, contact_base, exc, opts=None,
                           k1_values=None, k2_over_k1=0.85,
                           mode_index=0,
                           x_obs=0.5,
                           f_step=2000.0, n_points=200,
                           figsize=(13, 9), save_path=None):
    """
    Four-panel summary plot across a range of contact stiffnesses.

    (a) Normalised displacement  |z(x/L)|  at contact resonance
    (b) Normalised slope         |z'(x/L)| at contact resonance
    (c) Transfer-function amplitude  0–1 MHz  (log y-axis)  — measured at x_obs
    (d) Transfer-function phase      0–1 MHz

    Curves are colour-coded blue→red (plasma) from lowest to highest k₁.
    A marker (●) on panels (c)/(d) shows which resonance peak was used
    for the mode-shape solve.

    Parameters
    ----------
    cant, contact_base, exc : CantileverParams, ContactParams, ExcitationParams
    opts : SolverOptions, optional
        Used only to extract `g` and `method`; frequency range is overridden.
    k1_values : array-like, optional
        Contact stiffnesses to sweep [N/m].  Default: 6 log-spaced values
        from k/5 to k*2000.
    k2_over_k1 : float
        k2 = k2_over_k1 * k1 for each sweep point.
    mode_index : int
        Which eigenmode to use for panels (a)/(b), ranked by ascending
        frequency (0 = first/lowest, 1 = second, …).  If the requested
        mode is not found in the sweep window, falls back to the next
        available mode and prints a warning.
    x_obs : float
        Position along the cantilever (as fraction of L, 0–1) at which
        the displacement |z(x_obs·L)| is used for peak detection and
        plotted on panel (c).  Default 0.5 (mid-span).  Using the tip
        (x_obs=1) can miss the first mode at high k₁ because z(L)→0 as
        the contact stiffens toward the clamped-pinned limit.
    f_step : float
        Frequency step for the 0–1 MHz sweep [Hz].  Default 2000 Hz.
    n_points : int
        Spatial grid points for each BVP solve.
    """
    import matplotlib.colors as mcolors
    from .solver import frequency_sweep, solve_single_frequency, sho_analysis
    from .params import ContactParams, SolverOptions

    g      = opts.g      if opts is not None else 20.0
    method = opts.method if opts is not None else 'analytical'

    if k1_values is None:
        k1_values = np.logspace(np.log10(max(cant.k * 0.2, 0.1)),
                                np.log10(cant.k * 2000), 6)
    k1_values = np.asarray(k1_values, dtype=float)

    # Shared options for the 0–1 MHz sweep
    opts_sweep = SolverOptions(n_points=n_points, f_start=1e3, f_stop=1e6,
                               f_step=f_step, g=g, method=method)
    # Options for single-frequency mode-shape solve (f-range unused)
    opts_ms = SolverOptions(n_points=n_points, f_start=1e3, f_stop=1e6,
                            f_step=f_step, g=g, method=method)

    # Log-normalised colors so equal visual spacing across decades
    log_k1 = np.log10(k1_values)
    c_norm = (log_k1 - log_k1.min()) / max(log_k1.max() - log_k1.min(), 1e-12)
    colors = plt.cm.plasma(0.1 + 0.8 * c_norm)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_z, ax_zp   = axes[0, 0], axes[0, 1]
    ax_amp, ax_ph = axes[1, 0], axes[1, 1]

    for color, k1 in zip(colors, k1_values):
        mod_c = ContactParams(
            k1=k1, k2=k2_over_k1 * k1,
            c1=contact_base.c1, c2=k2_over_k1 * contact_base.c1,
            psi=contact_base.psi)

        # --- 0–1 MHz transfer function ---
        sweep = frequency_sweep(cant, mod_c, exc, opts_sweep)
        freqs_MHz = sweep['freqs'] / 1e6

        # Amplitude at observation point x_obs (fraction of L).
        # Using an interior point avoids the issue at high k₁ where z(L)→0
        # (clamped-pinned limit) and the first-mode peak vanishes in |z'(L)|.
        x_grid = sweep['results'][0]['x']
        obs_idx = int(np.argmin(np.abs(x_grid - x_obs * cant.contact_x)))
        amps_obs = np.array([np.abs(r['z'][obs_idx]) for r in sweep['results']])
        sweep_obs = dict(sweep, amplitudes=amps_obs)

        ax_amp.semilogy(freqs_MHz, amps_obs, color=color, lw=1.0, alpha=0.85)
        ax_ph.plot(freqs_MHz, sweep['phases'], color=color, lw=1.0, alpha=0.85)

        # --- mode shape at requested eigenmode (sorted by frequency) ---
        # Use find_peaks directly on the sampled amplitude array so that f_res
        # always lands on an actual data point. Bypassing SHO curve_fit avoids
        # the risk of the fitted frequency drifting between two resonances (where
        # modal interference would create spurious nodes in the mode shape).
        from scipy.signal import find_peaks as _find_peaks
        peak_idxs, _ = _find_peaks(amps_obs)
        if len(peak_idxs) == 0:
            peak_idxs = np.array([int(np.argmax(amps_obs))])
        peak_idxs_sorted = peak_idxs[np.argsort(sweep['freqs'][peak_idxs])]
        if mode_index < len(peak_idxs_sorted):
            f_res = sweep['freqs'][peak_idxs_sorted[mode_index]]
        else:
            print(f"  k1={k1:.1f} N/m: mode_index={mode_index} not found "
                  f"({len(peak_idxs_sorted)} peak(s) detected), using mode 0.")
            f_res = sweep['freqs'][peak_idxs_sorted[0]]

        # Mark selected resonance on TF panels
        a_at_res  = float(np.interp(f_res, sweep['freqs'], amps_obs))
        ph_at_res = float(np.interp(f_res, sweep['freqs'], sweep['phases']))
        ax_amp.plot(f_res / 1e6, a_at_res, 'o', color=color,
                    ms=4, zorder=5, markeredgewidth=0)
        ax_ph.plot(f_res / 1e6, ph_at_res, 'o', color=color,
                   ms=4, zorder=5, markeredgewidth=0)

        res = solve_single_frequency(f_res, cant, mod_c, exc, opts_ms)
        x_norm = res['x'] / cant.contact_x
        z_abs  = np.abs(res['z'])
        zp_abs = np.abs(res['zp'])

        if z_abs.max() > 0:
            ax_z.plot(x_norm, z_abs / z_abs.max(),
                      color=color, lw=1.2, alpha=0.85)
        if zp_abs.max() > 0:
            ax_zp.plot(x_norm, zp_abs / zp_abs.max(),
                       color=color, lw=1.2, alpha=0.85)

    # --- axis formatting ---
    _mn = f"mode {mode_index + 1}"
    panel_labels = [
        (ax_z,   'a', f'Displacement  |z(x/L)|  ({_mn}, norm.)',  'x / L',           '|z|  (norm.)'),
        (ax_zp,  'b', f"Slope  |z′(x/L)|  ({_mn}, norm.)",       'x / L',           "|z′|  (norm.)"),
        (ax_amp, 'c', f"Transfer function  |z({x_obs:.2g}·L)|",    'Frequency (MHz)',  f"|z({x_obs:.2g}·L)|"),
        (ax_ph,  'd', "Transfer function  phase",                  'Frequency (MHz)', 'Phase (°)'),
    ]
    for ax, letter, title, xlabel, ylabel in panel_labels:
        ax.set_title(f'({letter})  {title}', loc='left', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax_ph.set_ylim(-185, 185)
    ax_ph.set_yticks([-180, -90, 0, 90, 180])

    for ax in (ax_z, ax_zp):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.axvline(x_obs, color='0.5', lw=0.8, ls='--',
                   label=f'x_obs = {x_obs:.2g}·L')

    # --- inset colorbar in bottom-right corner of panel (b) ---
    sm = plt.cm.ScalarMappable(
        cmap='plasma',
        norm=mcolors.LogNorm(vmin=k1_values[0], vmax=k1_values[-1]))
    sm.set_array([])
    # inset_axes coords: [left, bottom, width, height] in axes-fraction units
    cax = ax_zp.inset_axes([0.62, 0.14, 0.34, 0.06])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('k₁  (N/m)', fontsize=9, labelpad=2)
    cbar.ax.tick_params(labelsize=7)
    # white background so the bar is readable over plot lines
    cbar.ax.set_facecolor('white')
    cbar.outline.set_edgecolor('0.4')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sho_check(sweep_result: dict, modes: list = None,
                   log_scale: bool = False,
                   figsize: tuple = (10, 5),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Diagnostic plot: amplitude spectrum with SHO fits overlaid.

    Useful for visually verifying that sho_analysis found the right peaks
    and that the fitted Q values are physically reasonable.

    Parameters
    ----------
    sweep_result : dict
        Output of frequency_sweep().
    modes : list, optional
        Output of sho_analysis(sweep_result).  If None, sho_analysis is run
        automatically with default settings.
    log_scale : bool
        If True, use a log y-axis for amplitude.
    """
    from .solver import sho_analysis

    freqs = sweep_result['freqs']
    amps  = sweep_result['amplitudes']
    freqs_kHz = freqs / 1e3

    if modes is None:
        modes = sho_analysis(sweep_result)

    def _sho_amp(f, f0, Q, A_res):
        x = f / f0
        return A_res / np.sqrt((Q * (1.0 - x * x)) ** 2 + x * x)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(freqs_kHz, amps, 'k-', lw=1.5, label='Simulated spectrum', zorder=2)

    colors = plt.cm.tab10(np.linspace(0, 0.7, max(len(modes), 1)))
    for i, m in enumerate(modes):
        f0   = m['f_res']
        Q    = m['Q']
        Ares = m['A_res']
        conv = m['converged']
        color = colors[i]

        # Overlay SHO curve
        sho_curve = _sho_amp(freqs, f0, Q, Ares)
        status = 'fit' if conv else 'bw-est'
        label = f"Mode {i+1}: f={f0/1e3:.2f} kHz  Q={Q:.1f} ({status})"
        ax.plot(freqs_kHz, sho_curve, '--', color=color, lw=1.2,
                label=label, zorder=3)

        # Mark the resonance frequency
        ax.axvline(f0 / 1e3, color=color, lw=0.8, alpha=0.6, zorder=1)
        ax.annotate(f"Q={Q:.0f}", xy=(f0 / 1e3, Ares),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=8, color=color)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude |z'(L)|")
    ax.set_title("SHO fit check")
    if log_scale:
        ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_spectra_slices(sf_map, k1_indices=None, n_slices=8,
                        figsize=(12, 6), save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    freqs_kHz = sf_map["freqs"] / 1e3
    k1        = sf_map["k1"]
    amp       = sf_map["amp_map"]

    if k1_indices is None:
        k1_indices = np.round(np.linspace(0, len(k1)-1, n_slices)).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    colors = cm.plasma(np.linspace(0, 1, len(k1_indices)))
    for color, ik in zip(colors, k1_indices):
        ax.semilogy(freqs_kHz, amp[ik], color=color, lw=1.2,
                    label="k1=%.2g N/m" % k1[ik])

    ax.set_xlabel("Frequency (kHz)", fontsize=12)
    ax.set_ylabel("Displacement Amplitude (m)", fontsize=12)
    ax.set_title("Contact Resonance Spectra - k1 sweep", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
