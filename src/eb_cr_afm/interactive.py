"""
Interactive Plotly visualizations for EB contact-resonance AFM results.

Each function returns a plotly.graph_objects.Figure that can be:
  - Displayed inline in Jupyter:  fig.show()
  - Saved as HTML:                fig.write_html("output.html")
  - Saved as PNG (requires kaleido): fig.write_image("output.png")
"""

from __future__ import annotations
import numpy as np


def plot_transfer_function_plotly(sweep_result: dict,
                                   title: str = "Transfer Function") :
    """
    Interactive amplitude + phase vs frequency plot.

    Parameters
    ----------
    sweep_result : dict
        Output of frequency_sweep().
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
        Two-panel figure: amplitude (pm/rad) on top, phase (°) on bottom.
        Hover shows frequency (kHz), amplitude, and phase.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    freqs_kHz = sweep_result["freqs"] / 1e3
    amps_pm   = sweep_result["amplitudes"] * 1e12
    phases    = sweep_result["phases"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Amplitude", "Phase"),
    )

    fig.add_trace(
        go.Scatter(
            x=freqs_kHz, y=amps_pm,
            mode="lines",
            name="Amplitude",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="f = %{x:.2f} kHz<br>A = %{y:.3f} pm/rad<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs_kHz, y=phases,
            mode="lines",
            name="Phase",
            line=dict(color="#d62728", width=2),
            hovertemplate="f = %{x:.2f} kHz<br>φ = %{y:.1f}°<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Frequency (kHz)", row=2, col=1)
    fig.update_yaxes(title_text="|z'(L)| (pm/rad)", row=1, col=1)
    fig.update_yaxes(title_text="Phase ∠z'(L) (°)", row=2, col=1)
    fig.update_layout(
        title=title,
        height=500,
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_displacement_profile_plotly(res: dict,
                                      title: str = "Cantilever Profile"):
    """
    Interactive mode-shape plot: displacement z(x) and slope z'(x).

    Parameters
    ----------
    res : dict
        Output of solve_single_frequency().
    title : str
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
        Two-panel figure (shared x-axis):
          Top:    z(x) — transverse displacement [pm].
                  Re (solid), Im (dashed), |z| (grey fill).
          Bottom: z'(x) = dz/dx — local slope angle [–, dimensionless].
                  Re (solid), Im (dashed), |z'| (grey fill).
                  Vertical dotted line at x = L marks the tip position
                  with an annotation showing the tip amplitude A = |z'(L)|.

        Physical interpretation:
          z(x)  — beam deflection profile (the mode shape)
          z'(x) — slope profile; z'(L) is the quantity detected by the
                  optical lever and reported as the observable amplitude A.
          z''(x) × EI = M(x)  (bending moment, available in res['zpp'])
          z'''(x) × (−EI) = Q(x)  (shear force, available in res['zppp'])
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_um     = res["x"] * 1e6
    L_um     = x_um[-1]
    freq_kHz = res["freq"] / 1e3

    z_pm  = res["z"]  * 1e12          # displacement in pm
    zp    = res["zp"]                  # slope, dimensionless

    A_tip = float(np.abs(zp[-1]))      # |z'(L)|

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Displacement  z(x)  (pm)",
            "Slope  z′(x) = dz/dx  (–)",
        ),
    )

    # ---- helper: add amplitude fill + Re + Im traces ----
    def _add_panel(row, y_complex, color_re, color_im, key_name, unit_str):
        y_abs = np.abs(y_complex)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_um, x_um[::-1]]),
                y=np.concatenate([y_abs, np.zeros(len(x_um))]),
                fill="toself",
                fillcolor="rgba(180,180,180,0.25)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=(row == 1),
                name="|amplitude|",
                hoverinfo="skip",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_um, y=np.real(y_complex),
                mode="lines", name=f"Re {key_name}",
                line=dict(color=color_re, width=2),
                hovertemplate=f"x = %{{x:.1f}} µm<br>Re = %{{y:.3g}} {unit_str}<extra></extra>",
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_um, y=np.imag(y_complex),
                mode="lines", name=f"Im {key_name}",
                line=dict(color=color_im, width=2, dash="dash"),
                hovertemplate=f"x = %{{x:.1f}} µm<br>Im = %{{y:.3g}} {unit_str}<extra></extra>",
            ),
            row=row, col=1,
        )

    _add_panel(1, z_pm, "#1f77b4", "#aec7e8", "z",  "pm")
    _add_panel(2, zp,   "#2ca02c", "#98df8a", "z′", "")

    # ---- tip marker on slope panel ----
    fig.add_vline(
        x=L_um,
        line=dict(color="#d62728", width=1.5, dash="dot"),
        row=2, col=1,
    )
    fig.add_annotation(
        x=L_um, y=A_tip,
        text=f"  A = {A_tip:.3g}",
        showarrow=False,
        font=dict(color="#d62728", size=11),
        xanchor="left",
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Position x (µm)", row=2, col=1)
    fig.update_yaxes(title_text="z(x)  (pm)",  row=1, col=1)
    fig.update_yaxes(title_text="z′(x)  (–)",  row=2, col=1)

    fig.update_layout(
        title=f"{title}  —  f = {freq_kHz:.2f} kHz",
        height=520,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_stiffness_sweep_plotly(sweep_result: dict):
    """
    Interactive plot of contact resonance frequency and Q factor vs k1.

    If sweep_result contains an 'all_modes' key (produced by contact_stiffness_sweep
    with sho_analysis), each eigenmode found in the frequency window is drawn as a
    separate trace, colour-coded by mode number.  Mode indices are assigned by
    ascending resonance frequency so they track continuously across k1.

    Parameters
    ----------
    sweep_result : dict
        Output of contact_stiffness_sweep().

    Returns
    -------
    plotly.graph_objects.Figure
        Two-panel figure (log k1 axis): f_res (kHz) left, Q right.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _COLORS = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
        "#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    k1        = sweep_result["k1"]
    all_modes = sweep_result.get("all_modes")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Contact Resonance Frequency", "Quality Factor"),
        horizontal_spacing=0.14,
    )

    if all_modes is not None:
        n_modes = max((len(m) for m in all_modes), default=1)
        for mode_idx in range(n_modes):
            k1_m, f_m, Q_m = [], [], []
            for ik, modes in enumerate(all_modes):
                if mode_idx < len(modes):
                    k1_m.append(k1[ik])
                    f_m.append(modes[mode_idx]['f_res'] / 1e3)
                    q_val = modes[mode_idx]['Q']
                    Q_m.append(q_val if np.isfinite(q_val) else float('nan'))

            color = _COLORS[mode_idx % len(_COLORS)]
            label = f"Mode {mode_idx + 1}"

            fig.add_trace(
                go.Scatter(
                    x=k1_m, y=f_m, mode="lines+markers", name=label,
                    marker=dict(size=5, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        "k₁ = %{x:.1f} N/m<br>"
                        "f = %{y:.2f} kHz"
                        f"<extra>{label}</extra>"
                    ),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=k1_m, y=Q_m, mode="lines+markers", name=label,
                    showlegend=False,
                    marker=dict(size=5, color=color),
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        "k₁ = %{x:.1f} N/m<br>"
                        "Q = %{y:.1f}"
                        f"<extra>{label}</extra>"
                    ),
                ),
                row=1, col=2,
            )
    else:
        # Backward-compatible single-mode path
        f_res = sweep_result["f_res"] / 1e3
        Q     = sweep_result["Q"]
        fig.add_trace(
            go.Scatter(
                x=k1, y=f_res, mode="lines+markers", name="f_res",
                marker=dict(size=5), line=dict(color=_COLORS[0], width=2),
                hovertemplate="k₁ = %{x:.1f} N/m<br>f_res = %{y:.2f} kHz<extra></extra>",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=k1, y=Q, mode="lines+markers", name="Q",
                marker=dict(size=5), line=dict(color=_COLORS[1], width=2),
                hovertemplate="k₁ = %{x:.1f} N/m<br>Q = %{y:.1f}<extra></extra>",
            ),
            row=1, col=2,
        )

    for col in (1, 2):
        fig.update_xaxes(title_text="k₁ (N/m)", type="log", row=1, col=col)
    fig.update_yaxes(title_text="f_res (kHz)", row=1, col=1)
    fig.update_yaxes(title_text="Q factor",    row=1, col=2)

    fig.update_layout(
        height=420,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_stiffness_map_plotly(sf_map: dict, cant=None):
    """
    Interactive 2D heatmap of amplitude and phase vs (frequency, k1).

    Parameters
    ----------
    sf_map : dict
        Output of compute_stiffness_frequency_map().
    cant : CantileverParams, optional
        If provided, adds horizontal guide lines for free / soft / stiff contact.

    Returns
    -------
    plotly.graph_objects.Figure
        Side-by-side heatmaps (log-scale k1 axis).
        Hover shows f (kHz), k1 (N/m), and amplitude or phase.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    freqs_kHz = sf_map["freqs"] / 1e3
    k1        = sf_map["k1"]
    amp       = sf_map["amp_map"]          # shape (Nk, Nf)
    phase     = sf_map["phase_map"]

    log_k1    = np.log10(k1)
    amp_log   = np.log10(np.clip(amp, 1e-17, None))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Amplitude (log₁₀ m)", "Phase (°)"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            x=freqs_kHz,
            y=log_k1,
            z=amp_log,
            colorscale="Plasma",
            colorbar=dict(title="log₁₀(m)", x=0.46, len=0.9),
            hovertemplate=(
                "f = %{x:.2f} kHz<br>"
                "k₁ = 10^%{y:.2f} N/m<br>"
                "log₁₀(A) = %{z:.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=freqs_kHz,
            y=log_k1,
            z=phase,
            colorscale="RdBu",
            zmin=-180, zmax=180,
            colorbar=dict(title="Phase (°)", x=1.01, len=0.9),
            hovertemplate=(
                "f = %{x:.2f} kHz<br>"
                "k₁ = 10^%{y:.2f} N/m<br>"
                "Phase = %{z:.1f}°<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    # Guide lines for free / soft / stiff contact
    if cant is not None:
        k = cant.k
        guides = [
            (np.log10(0.05 * k), "Free"),
            (np.log10(k),        "k₁ = k"),
            (np.log10(100 * k),  "k₁ = 100k"),
        ]
        for log_val, label in guides:
            if log_k1.min() <= log_val <= log_k1.max():
                for col in (1, 2):
                    fig.add_hline(
                        y=log_val,
                        line=dict(color="white", width=1, dash="dash"),
                        annotation_text=label,
                        annotation_font_color="white",
                        row=1, col=col,
                    )

    tick_vals = np.arange(np.floor(log_k1.min()), np.ceil(log_k1.max()) + 1)
    tick_text = [f"10<sup>{int(v)}</sup>" for v in tick_vals]

    for col in (1, 2):
        fig.update_xaxes(title_text="Frequency (kHz)", row=1, col=col)
        fig.update_yaxes(
            title_text="k₁ (N/m)",
            tickvals=tick_vals, ticktext=tick_text,
            row=1, col=col,
        )

    fig.update_layout(
        height=480,
        template="plotly_white",
    )
    return fig
