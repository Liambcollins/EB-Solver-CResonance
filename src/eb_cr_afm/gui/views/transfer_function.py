"""
Transfer function view widget.

Shows the amplitude and phase (°) of either the displacement z or the slope
z' (the OBD signal) at a chosen observation point vs frequency (kHz), with a
draggable cursor, SHO resonance-peak markers, and log/linear toggles.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QPushButton, QSplitter,
    QComboBox, QDoubleSpinBox, QLabel,
)

from ..branding import Palette
from ...solver import sho_analysis

# Log-mode floor: must be positive and well below any physical signal.
# z'(L) for typical CR-AFM with d1=1 pm is ~1e-9 to 1e-6 (dimensionless).
_LOG_FLOOR = 1e-15


class TransferFunctionView(QWidget):
    """
    Two stacked plots: amplitude (top) and phase (bottom).

    Signals
    -------
    frequency_selected(float)
        Emitted with frequency in Hz when the user clicks or drags the
        vertical cursor.
    """

    frequency_selected = Signal(float)
    obs_settings_changed = Signal()   # emitted when obs *position* changes (triggers recompute)
    sho_updated = Signal(object)      # list of fitted modes for the displayed quantity

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_freqs_khz: Optional[np.ndarray] = None
        self._current_amps: Optional[np.ndarray] = None
        self._last_sweep: Optional[dict] = None
        self._last_sho: Optional[list] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar row 1: log toggles
        tb = QHBoxLayout()
        self._log_amp_cb = QCheckBox("Log amplitude")
        self._log_freq_cb = QCheckBox("Log frequency")
        self._auto_range_btn = QPushButton("Auto range")

        tb.addWidget(self._log_amp_cb)
        tb.addWidget(self._log_freq_cb)
        tb.addWidget(self._auto_range_btn)
        tb.addStretch(1)
        layout.addLayout(tb)

        # Toolbar row 2: observation position + quantity
        obs_row = QHBoxLayout()
        obs_row.addWidget(QLabel("Obs. position:"))
        self._obs_pos_spin = QDoubleSpinBox()
        self._obs_pos_spin.setRange(0.0, 1000.0)
        self._obs_pos_spin.setValue(14.0)
        self._obs_pos_spin.setSuffix(" µm from tip")
        self._obs_pos_spin.setDecimals(1)
        self._obs_pos_spin.setFixedWidth(160)
        obs_row.addWidget(self._obs_pos_spin)
        obs_row.addWidget(QLabel("Quantity:"))
        self._quantity_combo = QComboBox()
        self._quantity_combo.addItems(["Displacement |z|", "Slope |z'|"])
        obs_row.addWidget(self._quantity_combo)
        obs_row.addStretch(1)
        layout.addLayout(obs_row)

        self._obs_pos_spin.valueChanged.connect(self._on_obs_changed)
        # Quantity change: re-render from cached data (no recompute needed)
        self._quantity_combo.currentIndexChanged.connect(self._on_quantity_changed)

        # Splitter with two PlotWidgets
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._amp_plot = pg.PlotWidget(title="Amplitude")
        self._phase_plot = pg.PlotWidget(title="Phase")

        # Do NOT use the `units=` kwarg — it triggers pyqtgraph's SI
        # auto-prefix (e.g. "nano", "mega") which garbles the axis labels.
        self._amp_plot.setLabel("bottom", "Frequency (kHz)")
        self._phase_plot.setLabel("bottom", "Frequency (kHz)")
        self._update_axis_labels()

        splitter.addWidget(self._amp_plot)
        splitter.addWidget(self._phase_plot)
        splitter.setSizes([300, 200])
        layout.addWidget(splitter)

        # Main data curves
        self._amp_curve = self._amp_plot.plot(pen=pg.mkPen(Palette.C_BLUE, width=2))
        self._phase_curve = self._phase_plot.plot(pen=pg.mkPen(Palette.C_PURPLE, width=2))

        for _pw in (self._amp_plot, self._phase_plot):
            _pw.showGrid(x=True, y=True, alpha=0.18)

        # Peak markers as PlotDataItems (not ScatterPlotItem) so that
        # pyqtgraph applies log-mode transformation to them automatically.
        # ScatterPlotItems added via addItem() are excluded from the
        # PlotItem's log-mode machinery and cause autoRange() to blow up.
        self._peak_markers_amp = self._amp_plot.plot(
            [], [], pen=None,
            symbol="t", symbolSize=10,
            symbolPen=pg.mkPen("r", width=1.5), symbolBrush=pg.mkBrush("r"),
        )
        self._peak_markers_phase = self._phase_plot.plot(
            [], [], pen=None,
            symbol="t", symbolSize=10,
            symbolPen=pg.mkPen("r", width=1.5), symbolBrush=pg.mkBrush("r"),
        )

        # SHO info text — start hidden so its default position (0, 0) does
        # not participate in auto-range calculations.
        self._sho_text = pg.TextItem(text="", color=Palette.INK, anchor=(0, 1))
        self._sho_text.setVisible(False)
        self._amp_plot.addItem(self._sho_text)

        # Vertical cursor lines — start hidden.
        # InfiniteLine.dataBounds(axis=0) returns (value, value), so a line
        # at the default pos=0 would force x=0 into the auto-range even when
        # data begins at 50 kHz.  We position them on the first data update.
        cursor_pen = pg.mkPen(Palette.C_CURSOR, width=1.4, style=Qt.PenStyle.DashLine)
        self._vline_amp = pg.InfiniteLine(
            angle=90, movable=True, pen=cursor_pen,
            label="", labelOpts={"position": 0.05, "color": Palette.C_CURSOR},
        )
        self._vline_phase = pg.InfiniteLine(
            angle=90, movable=True, pen=cursor_pen,
        )
        self._vline_amp.setVisible(False)
        self._vline_phase.setVisible(False)
        self._amp_plot.addItem(self._vline_amp)
        self._phase_plot.addItem(self._vline_phase)

        # Connect cursor drag
        self._vline_amp.sigPositionChanged.connect(self._on_cursor_moved_amp)
        self._vline_phase.sigPositionChanged.connect(self._on_cursor_moved_phase)

        # Connect toolbar
        self._log_amp_cb.stateChanged.connect(self._update_log_axes)
        self._log_freq_cb.stateChanged.connect(self._update_log_axes)
        self._auto_range_btn.clicked.connect(self._auto_range)

        # Connect mouse click on plots
        self._amp_plot.scene().sigMouseClicked.connect(self._on_amp_clicked)
        self._phase_plot.scene().sigMouseClicked.connect(self._on_phase_clicked)

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------

    def _cursor_freq_hz(self, vline: pg.InfiniteLine) -> float:
        """Return frequency in Hz from cursor position, accounting for log-x."""
        x = vline.value()
        if self._log_freq_cb.isChecked():
            # ViewBox x is in log10(kHz) when log-freq mode is active
            return (10 ** x) * 1e3
        return x * 1e3

    def _on_cursor_moved_amp(self):
        x_vb = self._vline_amp.value()
        self._vline_phase.blockSignals(True)
        self._vline_phase.setValue(x_vb)
        self._vline_phase.blockSignals(False)
        self.frequency_selected.emit(self._cursor_freq_hz(self._vline_amp))

    def _on_cursor_moved_phase(self):
        x_vb = self._vline_phase.value()
        self._vline_amp.blockSignals(True)
        self._vline_amp.setValue(x_vb)
        self._vline_amp.blockSignals(False)
        self.frequency_selected.emit(self._cursor_freq_hz(self._vline_phase))

    # ------------------------------------------------------------------
    # Mouse click
    # ------------------------------------------------------------------

    def _on_amp_clicked(self, event):
        pos = self._amp_plot.plotItem.vb.mapSceneToView(event.scenePos())
        x_vb = pos.x()
        self._vline_amp.setValue(x_vb)
        self._vline_phase.blockSignals(True)
        self._vline_phase.setValue(x_vb)
        self._vline_phase.blockSignals(False)
        self.frequency_selected.emit(self._cursor_freq_hz(self._vline_amp))

    def _on_phase_clicked(self, event):
        pos = self._phase_plot.plotItem.vb.mapSceneToView(event.scenePos())
        x_vb = pos.x()
        self._vline_phase.setValue(x_vb)
        self._vline_amp.blockSignals(True)
        self._vline_amp.setValue(x_vb)
        self._vline_amp.blockSignals(False)
        self.frequency_selected.emit(self._cursor_freq_hz(self._vline_phase))

    # ------------------------------------------------------------------
    # Log axis toggle
    # ------------------------------------------------------------------

    def _update_log_axes(self):
        log_amp = self._log_amp_cb.isChecked()
        log_freq = self._log_freq_cb.isChecked()
        self._amp_plot.setLogMode(x=log_freq, y=log_amp)
        self._phase_plot.setLogMode(x=log_freq, y=False)

        # Re-push amplitude data with appropriate floor for log mode.
        # pyqtgraph takes log10 of curve data; values ≤ 0 → -inf / NaN.
        if self._current_amps is not None and self._current_freqs_khz is not None:
            self._amp_curve.setData(
                self._current_freqs_khz,
                np.maximum(self._current_amps, _LOG_FLOOR),
            )

        # Reposition SHO text in the new coordinate space.
        self._reposition_sho_text(log_amp)

        # Re-range to fit the curve data (not the now-invalid old viewport).
        self._auto_range()

    def _reposition_sho_text(self, log_amp: bool):
        """Place _sho_text at the top-left of the amplitude data."""
        if not self._sho_text.isVisible() or self._current_amps is None:
            return
        y_max = float(np.max(self._current_amps))
        # When log-y is active the ViewBox coordinate system is in log10-space.
        # Passing y_max (a linear value) as the y-coordinate would place the
        # text at 10^y_max in linear space — exploding the auto-range.
        y_coord = float(np.log10(max(y_max, _LOG_FLOOR))) if log_amp else y_max
        x_coord = float(self._current_freqs_khz[0])
        if self._log_freq_cb.isChecked():
            x_coord = float(np.log10(max(x_coord, 1e-10)))
        self._sho_text.setPos(x_coord, y_coord)

    def _auto_range(self):
        if (self._current_freqs_khz is not None
                and len(self._current_freqs_khz) > 1):
            # Clamp x-axis to the actual data extent so log-freq mode doesn't
            # drift toward 0 or infinity via the InfiniteLine cursor.
            f0 = float(self._current_freqs_khz[0])
            f1 = float(self._current_freqs_khz[-1])
            pad = 0.02 * (f1 - f0)
            if self._log_freq_cb.isChecked():
                # ViewBox x-coords are log10(kHz) in log-freq mode
                x0 = np.log10(max(f0 - pad, 1e-3))
                x1 = np.log10(f1 + pad)
            else:
                x0, x1 = f0 - pad, f1 + pad
            self._amp_plot.setXRange(x0, x1, padding=0)
            self._phase_plot.setXRange(x0, x1, padding=0)
        self._amp_plot.autoRange(items=[self._amp_curve])
        self._phase_plot.autoRange(items=[self._phase_curve])

    # ------------------------------------------------------------------
    # Observation settings
    # ------------------------------------------------------------------

    def _on_obs_changed(self):
        """Observation *position* changed — trigger full recompute."""
        self._update_axis_labels()
        self.obs_settings_changed.emit()

    def _on_quantity_changed(self):
        """Quantity (disp/slope) changed — re-render from cached data."""
        self._update_axis_labels()
        if self._last_sweep is not None:
            # Re-fit the peaks on the newly selected observable.
            self.update_sweep(self._last_sweep)

    def _update_axis_labels(self):
        q = self.get_obs_quantity()
        if q == 'displacement':
            amp_label = "Displacement |z| (m)"
            phase_label = "Phase \u2220z (°)"
        else:
            amp_label = "Slope |z'| (rad)"
            phase_label = "Phase \u2220z' (°)"
        self._amp_plot.setLabel("left", amp_label)
        self._phase_plot.setLabel("left", phase_label)

    def get_obs_from_tip_m(self) -> float:
        """Observation distance from tip in metres."""
        return self._obs_pos_spin.value() * 1e-6

    def get_obs_quantity(self) -> str:
        return ('displacement' if self._quantity_combo.currentIndex() == 0
                else 'slope')

    # ------------------------------------------------------------------
    # Public: update data
    # ------------------------------------------------------------------

    def update_sweep(self, sweep_result: dict, sho_modes=None):
        """
        Update amplitude and phase curves.

        Parameters
        ----------
        sweep_result : dict
            Output of frequency_sweep().  Must contain both
            amplitudes_disp/phases_disp and amplitudes_slope/phases_slope.
        sho_modes : list or None
            Output of sho_analysis() for the displayed observable.  If None,
            the peaks are fitted here on the currently selected quantity.
        """
        if sho_modes is None:
            try:
                sho_modes = sho_analysis(sweep_result,
                                         quantity=self.get_obs_quantity())
            except Exception:
                sho_modes = []
        self._last_sweep = sweep_result
        self._last_sho   = sho_modes
        self.sho_updated.emit(sho_modes)

        freqs_hz = sweep_result["freqs"]

        # Select displayed quantity from the combo box
        if self.get_obs_quantity() == 'displacement':
            amps   = sweep_result.get("amplitudes_disp",  sweep_result["amplitudes"])
            phases = sweep_result.get("phases_disp",      sweep_result["phases"])
        else:
            amps   = sweep_result.get("amplitudes_slope", sweep_result["amplitudes"])
            phases = sweep_result.get("phases_slope",     sweep_result["phases"])

        freqs_khz = freqs_hz / 1e3
        self._current_freqs_khz = freqs_khz
        self._current_amps = amps

        log_amp = self._log_amp_cb.isChecked()
        # Always apply a tiny floor so autoRange never sees a zero-height range.
        self._amp_curve.setData(freqs_khz, np.maximum(amps, _LOG_FLOOR))
        self._phase_curve.setData(freqs_khz, phases)

        # Position cursor lines at midpoint on first data update
        if not self._vline_amp.isVisible() and len(freqs_khz) > 0:
            mid = float(freqs_khz[len(freqs_khz) // 2])
            if self._log_freq_cb.isChecked():
                mid = float(np.log10(max(mid, 1e-10)))
            self._vline_amp.setValue(mid)
            self._vline_phase.setValue(mid)
            self._vline_amp.setVisible(True)
            self._vline_phase.setVisible(True)

        # Clear old peak markers and text
        self._peak_markers_amp.setData([], [])
        self._peak_markers_phase.setData([], [])
        self._sho_text.setVisible(False)
        self._sho_text.setText("")

        if sho_modes:
            pk_x = [m["f_res"] / 1e3 for m in sho_modes]
            pk_amp = [m["A_res"] for m in sho_modes]
            pk_phase = [m["phase_res"] for m in sho_modes]

            self._peak_markers_amp.setData(pk_x, pk_amp)
            self._peak_markers_phase.setData(pk_x, pk_phase)

            lines = []
            for i, m in enumerate(sho_modes[:3]):
                conv_str = "" if m["converged"] else "*"
                lines.append(
                    f"Mode {i+1}: f={m['f_res']/1e3:.1f} kHz  "
                    f"Q={m['Q']:.0f}{conv_str}"
                )
            self._sho_text.setText("\n".join(lines))
            self._sho_text.setVisible(True)
            self._reposition_sho_text(log_amp)

        self._auto_range()
