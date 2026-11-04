"""
Mode shape (spatial profile) view widget.

Displays z(x) displacement and z'(x) slope profiles computed by
solve_single_frequency, with display mode (amplitude / real / imaginary)
and optional normalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QLabel, QSplitter,
)

from ..branding import Palette


class ModeShapeView(QWidget):
    """
    Two stacked pyqtgraph plots: z(x) on top, z'(x) on bottom.

    Signals
    -------
    solve_requested(float)
        Emitted when the user clicks 'Solve at freq' with the frequency [Hz].
    """

    solve_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[dict] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        tb = QHBoxLayout()

        tb.addWidget(QLabel("Display:"))
        self._display_combo = QComboBox()
        self._display_combo.addItems(["Amplitude", "Real", "Imaginary"])
        tb.addWidget(self._display_combo)

        tb.addWidget(QLabel("f (kHz):"))
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(0.1, 10000.0)
        self._freq_spin.setDecimals(2)
        self._freq_spin.setSuffix(" kHz")
        self._freq_spin.setValue(100.0)
        tb.addWidget(self._freq_spin)

        self._solve_btn = QPushButton("Solve at freq")
        tb.addWidget(self._solve_btn)

        self._normalize_cb = QCheckBox("Normalize")
        tb.addWidget(self._normalize_cb)

        tb.addStretch(1)
        layout.addLayout(tb)

        # Splitter with two PlotWidgets
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._z_plot = pg.PlotWidget(title="Displacement z(x)")
        self._zp_plot = pg.PlotWidget(title="Slope z'(x)")

        self._z_plot.setLabel("left", "z(x) (pm)")
        self._z_plot.setLabel("bottom", "Position (µm)")
        self._zp_plot.setLabel("left", "z'(x) (dimensionless)")
        self._zp_plot.setLabel("bottom", "Position (µm)")

        splitter.addWidget(self._z_plot)
        splitter.addWidget(self._zp_plot)
        splitter.setSizes([250, 250])
        layout.addWidget(splitter)

        for _pw in (self._z_plot, self._zp_plot):
            _pw.showGrid(x=True, y=True, alpha=0.18)

        # Curves: displacement
        self._z_curve = self._z_plot.plot(
            pen=pg.mkPen(Palette.C_BLUE, width=2), name="z(x)"
        )
        # Slope
        self._zp_curve = self._zp_plot.plot(
            pen=pg.mkPen(Palette.C_RED, width=2), name="z'(x)"
        )

        # Frequency label at top of z plot
        self._freq_text = pg.TextItem(text="", color=Palette.INK, anchor=(1, 0))
        self._z_plot.addItem(self._freq_text)

        # Wire signals
        self._display_combo.currentIndexChanged.connect(self._refresh)
        self._normalize_cb.stateChanged.connect(self._refresh)
        self._solve_btn.clicked.connect(self._on_solve_clicked)

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    def _on_solve_clicked(self):
        freq_hz = self._freq_spin.value() * 1e3
        self.solve_requested.emit(freq_hz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frequency(self, freq_hz: float):
        """Update the frequency spinbox (called from MainWindow cursor link)."""
        self._freq_spin.setValue(freq_hz / 1e3)

    def update_profile(self, result: dict):
        """
        Render new spatial profiles.

        Parameters
        ----------
        result : dict
            Output of solve_single_frequency().
        """
        self._result = result
        self._refresh()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _refresh(self):
        if self._result is None:
            return

        res = self._result
        x_um = res["x"] * 1e6   # m -> µm
        z = res["z"]
        zp = res["zp"]

        mode = self._display_combo.currentText()
        normalize = self._normalize_cb.isChecked()

        def _extract_z(arr):
            # z(x) is displacement in metres; convert to pm for display
            if mode == "Amplitude":
                return np.abs(arr) * 1e12
            elif mode == "Real":
                return np.real(arr) * 1e12
            else:
                return np.imag(arr) * 1e12

        def _extract_zp(arr):
            # z'(x) is dimensionless slope; keep as-is (no unit conversion needed)
            if mode == "Amplitude":
                return np.abs(arr)
            elif mode == "Real":
                return np.real(arr)
            else:
                return np.imag(arr)

        z_vals = _extract_z(z)
        zp_vals = _extract_zp(zp)

        if normalize:
            mx_z = np.max(np.abs(z_vals)) or 1.0
            mx_zp = np.max(np.abs(zp_vals)) or 1.0
            z_vals = z_vals / mx_z
            zp_vals = zp_vals / mx_zp
            self._z_plot.setLabel("left", "z(x) [norm.]")
            self._zp_plot.setLabel("left", "z'(x) [norm.]")
        else:
            self._z_plot.setLabel("left", "z(x) (pm)")
            self._zp_plot.setLabel("left", "z'(x) (dimensionless)")

        self._z_curve.setData(x_um, z_vals)
        self._zp_curve.setData(x_um, zp_vals)

        # Update frequency label
        freq_khz = res["freq"] / 1e3
        self._freq_text.setText(f"f = {freq_khz:.2f} kHz")
        if len(x_um) > 0:
            self._freq_text.setPos(x_um[-1], float(np.max(z_vals)))
