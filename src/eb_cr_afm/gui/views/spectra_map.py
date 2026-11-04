"""
Spectra / 2D map view widget.

Three modes (selected via the 'Map type' combo):
  - "Stiffness Map (k₁ vs f)": k₁ on y-axis vs frequency on x-axis;
    amplitude (left) and phase (right) side-by-side as heatmaps.
  - "Spatial Map (x vs f)": cantilever position x on y-axis vs frequency;
    colour = displacement |z(x)| or slope |z'(x)|.
  - "κ-Map (κ vs mode)": three stacked line-plot panels showing
    resonance frequency, OBD slope |z'(L)|, and tip displacement |z(L)|
    as functions of the dimensionless contact stiffness κ = k₁ L³ / EI.

A QStackedWidget swaps between the dual-image layout (modes 0/1) and the
3-panel line-plot layout (mode 2) so the existing image infrastructure is
preserved untouched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel,
    QProgressBar, QGroupBox, QFormLayout,
    QStackedWidget,
)
from PySide6.QtCore import QRectF

_CMAPS = ["inferno", "viridis", "plasma", "magma", "thermal", "CET-L8",
          "bipolar", "coolwarm", "greyclip"]
_CMAP_DEFAULT_AMP   = "inferno"
_CMAP_DEFAULT_PHASE = "bipolar"
_LOG_FLOOR = 1e-15

# Analytical zero-crossing positions (simplified model, H=mT=phi=0)
_KC_M1 = 28.44
_KC_M2 = 248.98

# Mode colours: Mode 1 blue, Mode 2 orange (matplotlib default)
_MODE_COLORS = ['#1f77b4', '#ff7f0e']


_SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3"
                    "\u2074\u2075\u2076\u2077\u2078\u2079")


class _Log10AxisItem(pg.AxisItem):
    """Axis that shows tick labels as 10^n when the data is in log10 space."""

    def tickStrings(self, values, scale, spacing):
        out = []
        for v in values:
            if abs(v - round(v)) < 0.05:
                exp_str = f"{int(round(v))}"
            else:
                exp_str = f"{v:.1f}"
            out.append("10" + exp_str.translate(_SUP))
        return out


class SpectraMapView(QWidget):
    """
    Image-based 2D map display with controls.

    Signals
    -------
    stiffness_map_requested(float, float, int, float, str)
        k1_min [N/m], k1_max [N/m], n_k1, obs_x [m], obs_quantity
    spatial_map_requested(int)
        n_x points
    kappa_map_requested(float, int, int)
        kappa_max, n_kappa, n_modes
    """

    # stiffness_map_requested: k1_min [N/m], k1_max [N/m], n_k1, obs_from_tip_m [m]
    # (obs_quantity is display-only; solver always computes both disp and slope)
    stiffness_map_requested = Signal(float, float, int, float)
    spatial_map_requested = Signal(int)
    kappa_map_requested = Signal(float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sf_map: Optional[dict] = None
        self._spatial_map: Optional[dict] = None
        self._kappa_map_data: Optional[dict] = None
        self._first_amp_image = True
        self._first_phase_image = True
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---- Mode selector ----
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Map type:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Stiffness Map (k\u2081 vs f)",
            "Spatial Map (x vs f)",
            "\u03ba-Map (\u03ba vs mode)",
        ])
        mode_row.addWidget(self._mode_combo, 1)
        layout.addLayout(mode_row)

        # ---- Controls group (image-map params) ----
        ctrl_grp = QGroupBox("Map Parameters")
        ctrl_layout = QHBoxLayout(ctrl_grp)

        form_left = QFormLayout()
        self._k1_min_spin = QDoubleSpinBox()
        self._k1_min_spin.setRange(0.1, 1e6)
        self._k1_min_spin.setValue(1.0)
        self._k1_min_spin.setSuffix(" N/m")
        self._k1_min_spin.setDecimals(2)

        self._k1_max_spin = QDoubleSpinBox()
        self._k1_max_spin.setRange(0.1, 1e6)
        self._k1_max_spin.setValue(10000.0)
        self._k1_max_spin.setSuffix(" N/m")
        self._k1_max_spin.setDecimals(1)

        self._n_k1_spin = QSpinBox()
        self._n_k1_spin.setRange(2, 500)
        self._n_k1_spin.setValue(40)

        self._n_x_spin = QSpinBox()
        self._n_x_spin.setRange(5, 500)
        self._n_x_spin.setValue(50)

        form_left.addRow("k\u2081 min:", self._k1_min_spin)
        form_left.addRow("k\u2081 max:", self._k1_max_spin)
        form_left.addRow("n k\u2081:", self._n_k1_spin)
        form_left.addRow("n x (spatial):", self._n_x_spin)

        form_right = QFormLayout()

        # Observation position (µm from tip)
        self._obs_pos_spin = QDoubleSpinBox()
        self._obs_pos_spin.setRange(0.0, 1000.0)
        self._obs_pos_spin.setValue(14.0)
        self._obs_pos_spin.setSuffix(" µm from tip")
        self._obs_pos_spin.setDecimals(1)

        # Quantity selector
        self._quantity_combo = QComboBox()
        self._quantity_combo.addItems(["Displacement |z|", "Slope |z'|"])

        # Display mode: single or dual channel
        self._display_combo = QComboBox()
        self._display_combo.addItems(["Amp + Phase", "Amplitude only", "Phase only"])

        # Amplitude colormap
        self._cmap_amp_combo = QComboBox()
        for name in _CMAPS:
            self._cmap_amp_combo.addItem(name)
        self._cmap_amp_combo.setCurrentText(_CMAP_DEFAULT_AMP)

        # Phase colormap
        self._cmap_phase_combo = QComboBox()
        for name in _CMAPS:
            self._cmap_phase_combo.addItem(name)
        self._cmap_phase_combo.setCurrentText(_CMAP_DEFAULT_PHASE)

        form_right.addRow("Obs. position:", self._obs_pos_spin)
        form_right.addRow("Quantity:", self._quantity_combo)
        form_right.addRow("Display:", self._display_combo)
        form_right.addRow("Amp colormap:", self._cmap_amp_combo)
        form_right.addRow("Phase colormap:", self._cmap_phase_combo)

        self._compute_btn = QPushButton("Compute")

        ctrl_layout.addLayout(form_left)
        ctrl_layout.addLayout(form_right)
        ctrl_layout.addWidget(self._compute_btn, 0, Qt.AlignmentFlag.AlignBottom)

        # ---- Controls group (κ-map params) ----
        kappa_grp = QGroupBox("\u03ba-Map Parameters")
        kappa_form = QFormLayout(kappa_grp)

        self._kappa_max_spin = QDoubleSpinBox()
        self._kappa_max_spin.setRange(10.0, 2000.0)
        self._kappa_max_spin.setValue(600.0)
        self._kappa_max_spin.setSingleStep(50.0)
        self._kappa_max_spin.setDecimals(0)

        self._n_kappa_spin = QSpinBox()
        self._n_kappa_spin.setRange(20, 500)
        self._n_kappa_spin.setValue(100)

        self._n_modes_combo = QComboBox()
        self._n_modes_combo.addItems(["1", "2"])
        self._n_modes_combo.setCurrentIndex(1)

        self._kappa_compute_btn = QPushButton("Compute \u03ba-Map")

        kappa_form.addRow("\u03ba max:", self._kappa_max_spin)
        kappa_form.addRow("n \u03ba points:", self._n_kappa_spin)
        kappa_form.addRow("n modes:", self._n_modes_combo)
        kappa_form.addRow("", self._kappa_compute_btn)

        # Start hidden; shown only in κ-Map mode
        kappa_grp.setVisible(False)
        self._kappa_grp = kappa_grp

        layout.addWidget(ctrl_grp)
        self._ctrl_grp = ctrl_grp
        layout.addWidget(kappa_grp)

        # ---- Progress bar ----
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # ---- Stacked widget: page 0 = image maps, page 1 = κ-map ----
        self._stack = QStackedWidget()

        # Page 0: two independent GL widgets so each can be shown/hidden
        # for single-channel vs dual-channel display.
        self._image_container = QWidget()
        _ic_layout = QHBoxLayout(self._image_container)
        _ic_layout.setContentsMargins(0, 0, 0, 0)
        _ic_layout.setSpacing(2)

        # --- Amplitude panel ---
        self._amp_gl = pg.GraphicsLayoutWidget()
        self._amp_plot = self._amp_gl.addPlot(
            row=0, col=0,
            axisItems={'left': _Log10AxisItem(orientation='left')})
        self._amp_plot.setLabel("bottom", "Frequency (kHz)")
        self._amp_plot.setLabel("left", "k\u2081 (N/m)")
        self._amp_plot.setTitle("Amplitude")
        self._amp_img = pg.ImageItem()
        self._amp_plot.addItem(self._amp_img)
        self._amp_hist = pg.HistogramLUTItem()
        self._amp_hist.setImageItem(self._amp_img)
        self._amp_gl.addItem(self._amp_hist, row=0, col=1)

        # --- Phase panel ---
        self._phase_gl = pg.GraphicsLayoutWidget()
        self._phase_plot = self._phase_gl.addPlot(
            row=0, col=0,
            axisItems={'left': _Log10AxisItem(orientation='left')})
        self._phase_plot.setLabel("bottom", "Frequency (kHz)")
        self._phase_plot.setLabel("left", "k\u2081 (N/m)")
        self._phase_plot.setTitle("Phase (\u00b0)")
        self._phase_img = pg.ImageItem()
        self._phase_plot.addItem(self._phase_img)
        self._phase_hist = pg.HistogramLUTItem()
        self._phase_hist.setImageItem(self._phase_img)
        self._phase_gl.addItem(self._phase_hist, row=0, col=1)

        _ic_layout.addWidget(self._amp_gl)
        _ic_layout.addWidget(self._phase_gl)

        self._apply_colormap(self._amp_hist, _CMAP_DEFAULT_AMP)
        self._apply_colormap(self._phase_hist, _CMAP_DEFAULT_PHASE)

        self._stack.addWidget(self._image_container)  # page 0

        # Page 1: κ-Map 3-panel line plot
        self._kappa_gl = pg.GraphicsLayoutWidget()

        # (a) Frequency vs κ
        self._kf_plot = self._kappa_gl.addPlot(row=0, col=0)
        self._kf_plot.setLabel("left", "Resonance freq (kHz)")
        self._kf_plot.setLabel("bottom", "\u03ba = k\u2081L\u00b3/EI")
        self._kf_plot.setTitle("(a)  Resonance frequency")
        self._kf_plot.addLegend(offset=(-10, 10))
        self._kf_plot.showGrid(x=True, y=True, alpha=0.3)

        # (b) |z'(L)| vs κ — log y
        self._kzp_plot = self._kappa_gl.addPlot(row=1, col=0)
        self._kzp_plot.setLabel("left", "|z\u2032(L)| (norm.)")
        self._kzp_plot.setLabel("bottom", "\u03ba = k\u2081L\u00b3/EI")
        self._kzp_plot.setTitle("(b)  OBD slope |z\u2032(L)|")
        self._kzp_plot.addLegend(offset=(-10, 10))
        self._kzp_plot.setLogMode(y=True)
        self._kzp_plot.showGrid(x=True, y=True, alpha=0.3)

        # (c) |z(L)| vs κ — log y
        self._kzl_plot = self._kappa_gl.addPlot(row=2, col=0)
        self._kzl_plot.setLabel("left", "|z(L)| / ref (norm.)")
        self._kzl_plot.setLabel("bottom", "\u03ba = k\u2081L\u00b3/EI")
        self._kzl_plot.setTitle("(c)  Tip displacement |z(L)|")
        self._kzl_plot.addLegend(offset=(-10, 10))
        self._kzl_plot.setLogMode(y=True)
        self._kzl_plot.showGrid(x=True, y=True, alpha=0.3)

        # Link x-axes so zoom/pan is synchronised
        self._kzp_plot.setXLink(self._kf_plot)
        self._kzl_plot.setXLink(self._kf_plot)

        self._stack.addWidget(self._kappa_gl)  # page 1

        layout.addWidget(self._stack, 1)

        # ---- Wire signals ----
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        self._kappa_compute_btn.clicked.connect(self._on_kappa_compute_clicked)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._cmap_amp_combo.currentTextChanged.connect(
            lambda name: self._apply_colormap(self._amp_hist, name))
        self._cmap_phase_combo.currentTextChanged.connect(
            lambda name: self._apply_colormap(self._phase_hist, name))
        # Quantity change: re-render from cached data (no recompute needed)
        self._quantity_combo.currentIndexChanged.connect(self._on_quantity_changed)
        # Display mode change: show/hide panels
        self._display_combo.currentIndexChanged.connect(self._update_image_visibility)

    # ------------------------------------------------------------------
    # Colormap
    # ------------------------------------------------------------------

    def _apply_colormap(self, hist: pg.HistogramLUTItem, name: str):
        cmap = None
        # Try pyqtgraph built-in first
        try:
            cmap = pg.colormap.get(name)
        except Exception:
            pass
        # Fall back to matplotlib (covers 'coolwarm' and many others)
        if cmap is None:
            try:
                cmap = pg.colormap.get(name, source='matplotlib')
            except Exception:
                pass
        if cmap is not None:
            hist.gradient.setColorMap(cmap)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_image_visibility(self):
        """Show/hide amp and phase panels based on the Display combo."""
        mode = self._display_combo.currentText()
        show_amp   = mode in ("Amp + Phase", "Amplitude only")
        show_phase = mode in ("Amp + Phase", "Phase only")
        self._amp_gl.setVisible(show_amp)
        self._phase_gl.setVisible(show_phase)

    def _on_mode_changed(self, idx: int):
        is_kappa = (idx == 2)
        is_stiffness = (idx == 0)

        # Show/hide parameter groups
        self._ctrl_grp.setVisible(not is_kappa)
        self._kappa_grp.setVisible(is_kappa)

        # Switch stacked pages
        self._stack.setCurrentIndex(1 if is_kappa else 0)

        if not is_kappa:
            self._k1_min_spin.setEnabled(is_stiffness)
            self._k1_max_spin.setEnabled(is_stiffness)
            self._n_k1_spin.setEnabled(is_stiffness)
            self._n_x_spin.setEnabled(not is_stiffness)
            y_label = "k\u2081 (N/m)" if is_stiffness else "Position (\u00b5m)"
            self._amp_plot.setLabel("left", y_label)
            self._phase_plot.setLabel("left", y_label)
            self._first_amp_image = True
            self._first_phase_image = True
            self._update_image_visibility()
            self._refresh_display()

        if is_kappa and self._kappa_map_data is not None:
            self._render_kappa_map()

    def _on_quantity_changed(self):
        """Quantity (disp/slope) changed — re-render from cached data."""
        self._first_amp_image = True
        self._first_phase_image = True
        self._refresh_display()

    def _on_compute_clicked(self):
        idx = self._mode_combo.currentIndex()
        if idx == 0:
            self.stiffness_map_requested.emit(
                self._k1_min_spin.value(),
                self._k1_max_spin.value(),
                self._n_k1_spin.value(),
                self._get_obs_x_m(),
            )
        elif idx == 1:
            self.spatial_map_requested.emit(self._n_x_spin.value())

    def _on_kappa_compute_clicked(self):
        kappa_max = self._kappa_max_spin.value()
        n_kappa = self._n_kappa_spin.value()
        n_modes = int(self._n_modes_combo.currentText())
        self.kappa_map_requested.emit(kappa_max, n_kappa, n_modes)

    def show_progress(self, pct: int):
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(pct)
        if pct >= 100:
            self._progress_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_stiffness_map(self, sf_map: dict):
        self._sf_map = sf_map
        self._mode_combo.setCurrentIndex(0)
        self._first_amp_image = True
        self._first_phase_image = True
        self._refresh_display()

    def update_spatial_map(self, spectra: dict):
        self._spatial_map = spectra
        self._mode_combo.setCurrentIndex(1)
        self._first_amp_image = True
        self._first_phase_image = True
        self._refresh_display()

    def update_kappa_map(self, km: dict):
        self._kappa_map_data = km
        # Switch to κ-Map mode so the result is immediately visible
        self._mode_combo.setCurrentIndex(2)
        self._render_kappa_map()

    # ------------------------------------------------------------------
    # Rendering helpers — image maps
    # ------------------------------------------------------------------

    def _set_image(self, img_item: pg.ImageItem, hist: pg.HistogramLUTItem,
                   data_2d: np.ndarray,
                   x_axis: np.ndarray, y_axis: np.ndarray,
                   first: bool) -> bool:
        """
        Push (Ny, Nx) data to an ImageItem and sync the histogram.

        Always auto-levels on new data so the colorscale adapts.
        The user can still drag the histogram handles for manual adjustment.
        Returns new value of 'first' (always False after first push).
        """
        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])
        img_data = data_2d.T.astype(np.float32)
        img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Always auto-scale so the histogram reflects the actual data range
        img_item.setImage(img_data, autoLevels=True)
        img_item.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        # Sync histogram range to the data so the gradient handles land in range
        vmin = float(np.nanmin(img_data))
        vmax = float(np.nanmax(img_data))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
            hist.setLevels(vmin, vmax)
        hist.autoHistogramRange()
        return False

    def _refresh_display(self):
        mode_idx = self._mode_combo.currentIndex()
        if mode_idx == 0:
            self._render_stiffness_map()
        elif mode_idx == 1:
            self._render_spatial_map()

    def _render_stiffness_map(self):
        if self._sf_map is None:
            return

        sf = self._sf_map
        freqs_khz = sf["freqs"] / 1e3
        log_k1 = np.log10(np.clip(sf["k1"], 1e-30, None))

        quantity = self._get_obs_quantity()
        if quantity == 'displacement':
            amp_key   = "amp_map_disp"
            phase_key = "phase_map_disp"
            q_label   = "Displacement |z|"
        else:
            amp_key   = "amp_map_slope"
            phase_key = "phase_map_slope"
            q_label   = "Slope |z'|"

        # Fall back to legacy keys if present (e.g. loaded from old NPZ)
        amp_raw   = sf.get(amp_key,   sf.get("amp_map",   None))
        phase_raw = sf.get(phase_key, sf.get("phase_map", None))
        if amp_raw is None or phase_raw is None:
            return

        amp_data = np.log10(np.clip(amp_raw.copy(), _LOG_FLOOR, None))
        self._first_amp_image = self._set_image(
            self._amp_img, self._amp_hist, amp_data,
            freqs_khz, log_k1, self._first_amp_image)
        self._amp_plot.setTitle(f"log\u2081\u2080 {q_label}")

        phase_data = phase_raw.copy()
        self._first_phase_image = self._set_image(
            self._phase_img, self._phase_hist, phase_data,
            freqs_khz, log_k1, self._first_phase_image)
        self._phase_plot.setTitle(f"Phase \u2220{q_label} (\u00b0)")

        for plot in (self._amp_plot, self._phase_plot):
            plot.setLabel("left", "k\u2081 (N/m)")

    def _render_spatial_map(self):
        if self._spatial_map is None:
            return

        sp = self._spatial_map
        freqs_khz = sp["freqs"] / 1e3
        x_um = sp["x_grid"] * 1e6

        if self._get_obs_quantity() == 'displacement':
            amp_raw   = sp["z_amp"]
            phase_raw = sp["z_phase"]
            amp_title   = "log\u2081\u2080 |z(x)| (pm)"
            phase_title = "Phase \u2220z(x) (\u00b0)"
        else:
            amp_raw   = sp["zp_amp"]
            phase_raw = sp["zp_phase"]
            amp_title   = "log\u2081\u2080 |z\u2032(x)|"
            phase_title = "Phase \u2220z\u2032(x) (\u00b0)"

        z_amp = np.log10(np.clip(amp_raw.copy() * 1e12, _LOG_FLOOR, None))
        self._first_amp_image = self._set_image(
            self._amp_img, self._amp_hist, z_amp,
            freqs_khz, x_um, self._first_amp_image)
        self._amp_plot.setTitle(amp_title)

        phase_data = phase_raw.copy()
        self._first_phase_image = self._set_image(
            self._phase_img, self._phase_hist, phase_data,
            freqs_khz, x_um, self._first_phase_image)
        self._phase_plot.setTitle(phase_title)

        for plot in (self._amp_plot, self._phase_plot):
            plot.setLabel("left", "Position (\u00b5m)")

    # ------------------------------------------------------------------
    # Rendering helpers — κ-map
    # ------------------------------------------------------------------

    def _render_kappa_map(self):
        if self._kappa_map_data is None:
            return

        km = self._kappa_map_data
        kappa   = np.array(km['kappa'], dtype=float)
        f_res   = km['f_res']
        zp_norm = km['zp_norm']
        zL_norm = km['zL_norm']
        f_free   = km.get('f_free', [])
        f_pinned = km.get('f_pinned', [])
        n_modes  = len(f_res)

        # Clear existing plot items
        self._kf_plot.clear()
        self._kzp_plot.clear()
        self._kzl_plot.clear()

        _vline_pen = pg.mkPen('gray', width=1.2,
                               style=Qt.PenStyle.DashLine)
        _kref_pen  = pg.mkPen('green', width=0.9,
                               style=Qt.PenStyle.DotLine)

        mode_names = ['Mode 1', 'Mode 2', 'Mode 3']

        for i in range(n_modes):
            color = _MODE_COLORS[i % len(_MODE_COLORS)]
            pen   = pg.mkPen(color, width=2)

            freq_khz = np.array(f_res[i]) / 1e3
            zp  = np.clip(np.array(zp_norm[i], dtype=float), 1e-6, None)
            zl  = np.clip(np.array(zL_norm[i], dtype=float), 1e-6, None)

            # (a) Frequency
            self._kf_plot.plot(kappa, freq_khz, pen=pen, name=mode_names[i])
            if i < len(f_free):
                self._kf_plot.addLine(
                    y=f_free[i] / 1e3,
                    pen=pg.mkPen(color, width=1,
                                 style=Qt.PenStyle.DotLine))
            if i < len(f_pinned):
                self._kf_plot.addLine(
                    y=f_pinned[i] / 1e3,
                    pen=pg.mkPen(color, width=1,
                                 style=Qt.PenStyle.DashDotLine))

            # (b) OBD slope
            self._kzp_plot.plot(kappa, zp, pen=pen, name=mode_names[i])

            # (c) Tip displacement
            self._kzl_plot.plot(kappa, zl, pen=pen, name=mode_names[i])

        # Vertical markers at analytical zero-crossing positions
        for plot in (self._kf_plot, self._kzp_plot, self._kzl_plot):
            plot.addLine(x=_KC_M1, pen=_vline_pen)
            if n_modes >= 2:
                plot.addLine(x=_KC_M2, pen=_vline_pen)
            # κ = 3 reference (k₁ = k_c for standard cantilevers)
            plot.addLine(x=3.0, pen=_kref_pen)

        # Reset x range to full κ span
        x_max = float(kappa[-1]) if len(kappa) else 600.0
        self._kf_plot.setXRange(0.0, x_max, padding=0.02)
        for plot in (self._kf_plot, self._kzp_plot, self._kzl_plot):
            plot.autoRange()

    # ------------------------------------------------------------------
    # Accessors for MainWindow
    # ------------------------------------------------------------------

    def get_k1_range(self):
        return (
            self._k1_min_spin.value(),
            self._k1_max_spin.value(),
            self._n_k1_spin.value(),
        )

    def get_n_x(self):
        return self._n_x_spin.value()

    def _get_obs_x_m(self) -> float:
        return self._obs_pos_spin.value() * 1e-6

    def _get_obs_quantity(self) -> str:
        return ('displacement' if self._quantity_combo.currentIndex() == 0
                else 'slope')

    def get_obs_settings(self):
        """Return (obs_from_tip_m, obs_quantity)."""
        return self._get_obs_x_m(), self._get_obs_quantity()

    def get_kappa_settings(self):
        """Return (kappa_max, n_kappa, n_modes)."""
        return (
            self._kappa_max_spin.value(),
            self._n_kappa_spin.value(),
            int(self._n_modes_combo.currentText()),
        )
