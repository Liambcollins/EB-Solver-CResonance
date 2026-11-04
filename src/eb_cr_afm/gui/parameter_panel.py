"""
Parameter panel widget for the CR2Beam GUI.

Provides grouped controls for **every** solver parameter, with preset loading,
bidirectional log sliders, tooltips, and JSON serialization.
"""

from __future__ import annotations

import json
import math
from typing import Tuple

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QDoubleSpinBox, QSpinBox,
    QSlider, QCheckBox, QFormLayout, QSizePolicy, QScrollArea,
    QFrame,
)

from eb_cr_afm.params import (
    CantileverParams, ContactParams, ExcitationParams, SolverOptions,
)
from eb_cr_afm import (default_params, default_params_multi75,
                       default_params_null_spot_paper)


# ---------------------------------------------------------------------------
# Log slider mapping  log10(k1) in [-1, 6] mapped to slider [0,1000]
# ---------------------------------------------------------------------------
_K1_LOG_MIN = -1.0   # log10(0.1)
_K1_LOG_MAX = 6.0    # log10(1e6)
_SLIDER_MAX = 1000


def _spin_to_stiff(val: float) -> float:
    """Spin-box value -> stiffness; 0 in the GUI means rigid (infinite)."""
    return math.inf if val <= 0.0 else float(val)


def _stiff_to_spin(k: float) -> float:
    """Stiffness -> spin-box value; infinite (rigid) is shown as 0."""
    return 0.0 if math.isinf(k) else float(k)


def _slider_to_k1(slider_val: int) -> float:
    frac = slider_val / _SLIDER_MAX
    log_k1 = _K1_LOG_MIN + frac * (_K1_LOG_MAX - _K1_LOG_MIN)
    return 10.0 ** log_k1


def _k1_to_slider(k1: float) -> int:
    k1 = max(10.0 ** _K1_LOG_MIN, min(10.0 ** _K1_LOG_MAX, k1))
    frac = (math.log10(k1) - _K1_LOG_MIN) / (_K1_LOG_MAX - _K1_LOG_MIN)
    return int(round(frac * _SLIDER_MAX))


# ---------------------------------------------------------------------------
# Log amplitude helper for drive amplitude (pm)
# ---------------------------------------------------------------------------
_AMP_LOG_MIN = math.log10(0.01)    # 0.01 pm
_AMP_LOG_MAX = math.log10(1000.0)  # 1000 pm


def _slider_to_amp(slider_val: int) -> float:
    if slider_val == 0:
        return 0.0
    frac = slider_val / _SLIDER_MAX
    return 10.0 ** (_AMP_LOG_MIN + frac * (_AMP_LOG_MAX - _AMP_LOG_MIN))


def _amp_to_slider(amp_pm: float) -> int:
    if amp_pm <= 0.0:
        return 0
    amp_pm = max(10.0 ** _AMP_LOG_MIN, min(10.0 ** _AMP_LOG_MAX, amp_pm))
    frac = (math.log10(amp_pm) - _AMP_LOG_MIN) / (_AMP_LOG_MAX - _AMP_LOG_MIN)
    return int(round(frac * _SLIDER_MAX))


class ParameterPanel(QWidget):
    """
    Left-panel widget exposing all solver parameters grouped in QGroupBoxes.

    Signals
    -------
    params_changed
        Emitted whenever any parameter control changes value.
    compute_tf_requested
        Emitted when the 'Compute Transfer Function' button is clicked.
    compute_map_requested
        Emitted when the 'Compute 2D Map' button is clicked.
    """

    params_changed = Signal()
    compute_tf_requested = Signal()
    compute_map_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block_signals = False

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        self._main_layout = QVBoxLayout(inner)
        self._main_layout.setSpacing(8)
        self._main_layout.setContentsMargins(6, 6, 6, 6)

        self._build_preset_group()
        self._build_cantilever_group()
        self._build_contact_group()
        self._build_drive_group()
        self._build_es_group()
        self._build_solver_group()
        self._build_compute_group()

        self._main_layout.addStretch(1)
        scroll.setWidget(inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Load default preset
        self._apply_preset()

    # ------------------------------------------------------------------
    # Group builders
    # ------------------------------------------------------------------

    def _build_preset_group(self):
        grp = QGroupBox("Preset")
        layout = QHBoxLayout(grp)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems([
            "Multi 75-G (default)",
            "Table I (Bradler 2016)",
            "Null-spot paper (ACS Nano 2026)",
        ])
        self._preset_combo.setToolTip("Load a built-in parameter set.")
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_preset)

        layout.addWidget(self._preset_combo, 1)
        layout.addWidget(self._apply_btn)
        self._main_layout.addWidget(grp)

    def _build_cantilever_group(self):
        grp = QGroupBox("Cantilever")
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._L_spin = self._make_dspin(50, 500, 1, suffix=" µm",
                                        tip="Cantilever length L.")
        self._w_spin = self._make_dspin(10, 100, 0.5, suffix=" µm",
                                        tip="Cantilever width w.")
        self._H_spin = self._make_dspin(0, 30, 0.1, suffix=" µm",
                                        tip="Tip height H (apex below the lever plane).")
        self._k_spin = self._make_dspin(0.01, 100, 0.01, suffix=" N/m", decimals=3,
                                        tip="Static spring constant k (EI = kL³/3).")
        self._phi_spin = self._make_dspin(0, 25, 0.1, suffix=" °",
                                          tip="Lever tilt angle φ.")
        self._m_spin = self._make_dspin(0.1, 5000, 0.1, suffix=" ng", decimals=3,
                                        tip="Total lever mass m (1 ng = 1e-12 kg).")
        self._mT_spin = self._make_dspin(0.0, 5000, 0.1, suffix=" ng", decimals=3,
                                         tip="Lumped tip mass m_T at H/4 (Rabe convention).")
        self._tip_offset_spin = self._make_dspin(0, 100, 1, suffix=" µm", decimals=1,
                                                 tip="Axial set-back of the contact from the "
                                                     "free end. >0 activates the two-segment "
                                                     "overhang model (contact_x = L − offset).")
        self._ow_ratio_spin = self._make_dspin(0.0, 1.0, 0.05, decimals=3,
                                               tip="Electrostatic width of the overhang as a "
                                                   "fraction of w (1 = full width). Affects only "
                                                   "the distributed electrostatic load.")

        form.addRow("L (length):", self._L_spin)
        form.addRow("w (width):", self._w_spin)
        form.addRow("H (tip height):", self._H_spin)
        form.addRow("k (spring const):", self._k_spin)
        form.addRow("φ (tilt angle):", self._phi_spin)
        form.addRow("m (lever mass):", self._m_spin)
        form.addRow("m_T (tip mass):", self._mT_spin)
        form.addRow("tip offset:", self._tip_offset_spin)
        form.addRow("overhang width:", self._ow_ratio_spin)

        for spin in (self._L_spin, self._w_spin, self._H_spin,
                     self._k_spin, self._phi_spin, self._m_spin,
                     self._mT_spin, self._tip_offset_spin, self._ow_ratio_spin):
            spin.valueChanged.connect(self._on_param_changed)

        self._main_layout.addWidget(grp)

    def _build_contact_group(self):
        grp = QGroupBox("Contact")
        layout = QVBoxLayout(grp)

        # k1 log slider + spinbox
        k1_row = QHBoxLayout()
        k1_lbl = QLabel("k₁ (N/m):")
        k1_lbl.setFixedWidth(80)
        self._k1_slider = QSlider(Qt.Orientation.Horizontal)
        self._k1_slider.setRange(0, _SLIDER_MAX)
        self._k1_slider.setToolTip("Vertical contact stiffness k₁ (log slider).")
        self._k1_spin = self._make_dspin(0.1, 1e6, 1.0, suffix=" N/m", decimals=3,
                                         tip="Vertical contact stiffness k₁.")

        k1_row.addWidget(k1_lbl)
        k1_row.addWidget(self._k1_slider, 2)
        k1_row.addWidget(self._k1_spin, 1)
        layout.addLayout(k1_row)

        # k2 with auto checkbox
        k2_row = QHBoxLayout()
        k2_lbl = QLabel("k₂ (N/m):")
        k2_lbl.setFixedWidth(80)
        self._k2_auto_cb = QCheckBox("Auto (0.85·k₁)")
        self._k2_auto_cb.setToolTip("Tie lateral stiffness to k₂ = 0.85·k₁ "
                                    "(Mindlin no-slip ratio).")
        self._k2_spin = self._make_dspin(0.0, 1e6, 1.0, suffix=" N/m", decimals=3,
                                         tip="Lateral (longitudinal) contact stiffness k₂.")

        k2_row.addWidget(k2_lbl)
        k2_row.addWidget(self._k2_auto_cb)
        k2_row.addWidget(self._k2_spin, 1)
        layout.addLayout(k2_row)

        # c1, c2, psi
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._c1_spin = self._make_dspin(0, 500, 0.1, suffix=" µN·s/m", decimals=3,
                                         tip="Vertical contact damping c₁.")
        self._c2_spin = self._make_dspin(0, 500, 0.1, suffix=" µN·s/m", decimals=3,
                                         tip="Longitudinal contact damping c₂.")
        self._psi_spin = self._make_dspin(-25, 25, 0.1, suffix=" °", decimals=2,
                                          tip="Local sample tilt ψ. Contact angle θ = φ + ψ.")
        form.addRow("c₁ (vert damp):", self._c1_spin)
        form.addRow("c₂ (long damp):", self._c2_spin)
        form.addRow("ψ (sample tilt):", self._psi_spin)
        self._kcl_spin = self._make_dspin(0.0, 1e7, 100.0, suffix=" N/m", decimals=1,
                                          tip="Lateral bending stiffness of the elastic tip "
                                              "cone at its apex. 0 = rigid tip (original "
                                              "Bradler model). Typically a few kN/m.")
        self._kcl_spin.setSpecialValueText("rigid (∞)")
        self._kca_spin = self._make_dspin(0.0, 1e9, 1000.0, suffix=" N/m", decimals=1,
                                          tip="Axial stiffness of the tip cone. 0 = rigid "
                                              "(recommended; usually far above k₁).")
        self._kca_spin.setSpecialValueText("rigid (∞)")
        form.addRow("tip cone k_lat:", self._kcl_spin)
        form.addRow("tip cone k_ax:", self._kca_spin)
        layout.addLayout(form)

        # Wire signals
        self._k1_slider.valueChanged.connect(self._on_k1_slider_changed)
        self._k1_spin.valueChanged.connect(self._on_k1_spin_changed)
        self._k2_auto_cb.stateChanged.connect(self._on_k2_auto_changed)
        self._k2_spin.valueChanged.connect(self._on_param_changed)
        self._c1_spin.valueChanged.connect(self._on_param_changed)
        self._c2_spin.valueChanged.connect(self._on_param_changed)
        self._psi_spin.valueChanged.connect(self._on_param_changed)
        self._kcl_spin.valueChanged.connect(self._on_param_changed)
        self._kca_spin.valueChanged.connect(self._on_param_changed)

        self._k2_auto_cb.setChecked(True)
        self._k2_spin.setEnabled(False)

        self._main_layout.addWidget(grp)

    def _build_drive_group(self):
        grp = QGroupBox("Mechanical Drive")
        layout = QVBoxLayout(grp)

        # Drive type
        drive_row = QHBoxLayout()
        drive_row.addWidget(QLabel("Drive type:"))
        self._drive_combo = QComboBox()
        self._drive_combo.addItems([
            "Vertical (d₁)",
            "Longitudinal (d₂)",
            "Both",
            "None (electrostatic only)",
        ])
        self._drive_combo.setToolTip("Which sample-displacement drive component is active.")
        drive_row.addWidget(self._drive_combo, 1)
        layout.addLayout(drive_row)

        # Amplitude log slider
        amp_row = QHBoxLayout()
        amp_lbl = QLabel("Amplitude:")
        amp_lbl.setFixedWidth(80)
        self._amp_slider = QSlider(Qt.Orientation.Horizontal)
        self._amp_slider.setRange(0, _SLIDER_MAX)
        self._amp_slider.setToolTip("Sample-displacement drive amplitude (log slider).")
        self._amp_spin = self._make_dspin(0.0, 1000.0, 0.01, suffix=" pm", decimals=4,
                                          tip="Sample-displacement drive amplitude d₁/d₂.")

        amp_row.addWidget(amp_lbl)
        amp_row.addWidget(self._amp_slider, 2)
        amp_row.addWidget(self._amp_spin, 1)
        layout.addLayout(amp_row)

        self._drive_combo.currentIndexChanged.connect(self._on_param_changed)
        self._amp_slider.valueChanged.connect(self._on_amp_slider_changed)
        self._amp_spin.valueChanged.connect(self._on_amp_spin_changed)

        self._main_layout.addWidget(grp)

    def _build_es_group(self):
        grp = QGroupBox("Electrostatic Forcing")
        layout = QVBoxLayout(grp)

        # Source toggles
        self._use_dist_load_cb = QCheckBox("Distributed load q(x)")
        self._use_dist_load_cb.setToolTip(
            "Include the distributed body-electrostatic load q(x) along the lever.")
        self._use_f0_cb = QCheckBox("Local tip force f₀")
        self._use_f0_cb.setToolTip("Include a local electrostatic force at the tip apex.")
        layout.addWidget(self._use_dist_load_cb)
        layout.addWidget(self._use_f0_cb)

        # Bias / medium form
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._vac_spin = self._make_dspin(0.0, 100.0, 0.1, suffix=" V", decimals=3,
                                          tip="AC bias amplitude V_AC.")
        self._dv_spin = self._make_dspin(-100.0, 100.0, 0.1, suffix=" V", decimals=3,
                                         tip="DC potential difference ΔV (applied DC + surface pot.).")
        self._eps_spin = self._make_dspin(1.0, 100.0, 0.1, suffix="", decimals=3,
                                          tip="Relative permittivity of the gap medium (air ≈ 1).")
        self._lift_spin = self._make_dspin(0.0, 100000.0, 5.0, suffix=" nm", decimals=1,
                                           tip="Rigid lift height of the lever above the sample "
                                               "(0 = contact mode).")
        form.addRow("V_AC:", self._vac_spin)
        form.addRow("ΔV (DC):", self._dv_spin)
        form.addRow("ε_r:", self._eps_spin)
        form.addRow("lift height:", self._lift_spin)
        layout.addLayout(form)

        # Local-force detail
        sub = QLabel("Local tip force")
        sub.setObjectName("SubHead")
        layout.addWidget(sub)

        form2 = QFormLayout()
        form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._f0_spin = self._make_dspin(0.0, 1e6, 1.0, suffix=" pN", decimals=3,
                                         tip="Local tip-force amplitude f₀ (used when 'Local "
                                             "tip force' is on and bias-coupling is off).")
        self._cg_spin = self._make_dspin(0.0, 1e5, 1.0, suffix=" nF/m", decimals=3,
                                         tip="Capacitance gradient dC/dz for the square-law "
                                             "bias-coupled tip force.")
        form2.addRow("f₀:", self._f0_spin)
        form2.addRow("dC/dz:", self._cg_spin)
        layout.addLayout(form2)

        self._bias_f0_cb = QCheckBox("Bias-coupled f₀ = ½(dC/dz)·V_bias²")
        self._bias_f0_cb.setToolTip(
            "Derive the local tip force from ½·(dC/dz)·(ΔV + V_AC)² instead of f₀.")
        layout.addWidget(self._bias_f0_cb)

        # Wire signals
        self._use_dist_load_cb.stateChanged.connect(self._on_param_changed)
        self._use_f0_cb.stateChanged.connect(self._on_param_changed)
        for w in (self._vac_spin, self._dv_spin, self._eps_spin,
                  self._lift_spin, self._f0_spin, self._cg_spin):
            w.valueChanged.connect(self._on_param_changed)
        self._bias_f0_cb.stateChanged.connect(self._on_param_changed)

        self._main_layout.addWidget(grp)

    def _build_solver_group(self):
        grp = QGroupBox("Solver")
        form = QFormLayout(grp)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["analytical", "bvp"])
        self._method_combo.setToolTip("Closed-form (fast) or scipy solve_bvp backend.")

        self._f_start_spin = self._make_dspin(0.1, 10000, 1.0, suffix=" kHz", decimals=1,
                                              tip="Sweep start frequency.")
        self._f_stop_spin = self._make_dspin(0.1, 10000, 1.0, suffix=" kHz", decimals=1,
                                             tip="Sweep stop frequency.")
        self._f_step_spin = self._make_dspin(1, 100000, 100, suffix=" Hz", decimals=0,
                                             tip="Frequency step (linewidth ~Hz at low g — "
                                                 "keep fine near resonance).")
        self._n_points_spin = QSpinBox()
        self._n_points_spin.setRange(10, 5000)
        self._n_points_spin.setValue(300)
        self._n_points_spin.setToolTip("Spatial grid points along the lever.")
        self._g_spin = self._make_dspin(0, 1e6, 100.0, suffix=" 1/s", decimals=1,
                                        tip="Air damping coefficient g (η in Bradler 2016). "
                                            "Table I uses 20 000 1/s.")

        form.addRow("Method:", self._method_combo)
        form.addRow("f start:", self._f_start_spin)
        form.addRow("f stop:", self._f_stop_spin)
        form.addRow("f step:", self._f_step_spin)
        form.addRow("n points:", self._n_points_spin)
        form.addRow("g (air damp):", self._g_spin)

        self._method_combo.currentIndexChanged.connect(self._on_param_changed)
        for w in (self._f_start_spin, self._f_stop_spin, self._f_step_spin,
                  self._g_spin):
            w.valueChanged.connect(self._on_param_changed)
        self._n_points_spin.valueChanged.connect(self._on_param_changed)

        self._main_layout.addWidget(grp)

    def _build_compute_group(self):
        grp = QGroupBox("Compute")
        layout = QVBoxLayout(grp)

        self._compute_tf_btn = QPushButton("Compute Transfer Function")
        self._compute_map_btn = QPushButton("Compute 2D Map")
        for b in (self._compute_tf_btn, self._compute_map_btn):
            b.setProperty("primary", True)

        self._compute_tf_btn.clicked.connect(self.compute_tf_requested)
        self._compute_map_btn.clicked.connect(self.compute_map_requested)

        layout.addWidget(self._compute_tf_btn)
        layout.addWidget(self._compute_map_btn)
        self._main_layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Widget factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dspin(vmin, vmax, step, suffix="", decimals=2, tip="") -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(vmin, vmax)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        if suffix:
            sb.setSuffix(suffix)
        if tip:
            sb.setToolTip(tip)
        sb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return sb

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_param_changed(self):
        if self._block_signals:
            return
        self.params_changed.emit()

    def _on_k1_slider_changed(self, val: int):
        if self._block_signals:
            return
        self._block_signals = True
        k1 = _slider_to_k1(val)
        self._k1_spin.setValue(k1)
        if self._k2_auto_cb.isChecked():
            self._k2_spin.setValue(0.85 * k1)
        self._block_signals = False
        self.params_changed.emit()

    def _on_k1_spin_changed(self, val: float):
        if self._block_signals:
            return
        self._block_signals = True
        self._k1_slider.setValue(_k1_to_slider(val))
        if self._k2_auto_cb.isChecked():
            self._k2_spin.setValue(0.85 * val)
        self._block_signals = False
        self.params_changed.emit()

    def _on_k2_auto_changed(self, state):
        auto = (state == Qt.CheckState.Checked.value or state == 2)
        self._k2_spin.setEnabled(not auto)
        if auto:
            k1 = self._k1_spin.value()
            self._block_signals = True
            self._k2_spin.setValue(0.85 * k1)
            self._block_signals = False
        self.params_changed.emit()

    def _on_amp_slider_changed(self, val: int):
        if self._block_signals:
            return
        self._block_signals = True
        self._amp_spin.setValue(_slider_to_amp(val))
        self._block_signals = False
        self.params_changed.emit()

    def _on_amp_spin_changed(self, val: float):
        if self._block_signals:
            return
        self._block_signals = True
        self._amp_slider.setValue(_amp_to_slider(val))
        self._block_signals = False
        self.params_changed.emit()

    def _apply_preset(self):
        idx = self._preset_combo.currentIndex()
        if idx == 0:
            cant, contact, exc, opts = default_params_multi75()
        elif idx == 1:
            cant, contact, exc, opts = default_params()
        else:
            cant, contact, exc, opts = default_params_null_spot_paper()
        self.set_params(cant, contact, exc, opts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> Tuple[CantileverParams, ContactParams,
                                  ExcitationParams, SolverOptions]:
        """Read current control values and return parameter dataclasses."""
        cant = CantileverParams(
            L=self._L_spin.value() * 1e-6,
            w=self._w_spin.value() * 1e-6,
            H=self._H_spin.value() * 1e-6,
            k=self._k_spin.value(),
            m=self._m_spin.value() * 1e-12,     # ng -> kg
            mT=self._mT_spin.value() * 1e-12,   # ng -> kg
            phi=math.radians(self._phi_spin.value()),
            tip_offset=self._tip_offset_spin.value() * 1e-6,
            overhang_width_ratio=self._ow_ratio_spin.value(),
        )

        k1 = self._k1_spin.value()
        k2 = self._k2_spin.value()
        c1 = self._c1_spin.value() * 1e-6   # µN·s/m -> N·s/m
        c2 = self._c2_spin.value() * 1e-6

        contact = ContactParams(
            k1=k1, k2=k2, c1=c1, c2=c2,
            psi=math.radians(self._psi_spin.value()),
            k_cone_lat=_spin_to_stiff(self._kcl_spin.value()),
            k_cone_ax=_spin_to_stiff(self._kca_spin.value()),
        )

        amp_m = self._amp_spin.value() * 1e-12  # pm -> m
        drive_idx = self._drive_combo.currentIndex()
        if drive_idx == 0:    # Vertical
            d1, d2 = amp_m, 0.0
        elif drive_idx == 1:  # Longitudinal
            d1, d2 = 0.0, amp_m
        elif drive_idx == 2:  # Both
            d1, d2 = amp_m, amp_m
        else:                 # None — electrostatic forcing only
            d1, d2 = 0.0, 0.0

        exc = ExcitationParams(
            d1=d1, d2=d2,
            f0=self._f0_spin.value() * 1e-12,          # pN -> N
            delta_V=self._dv_spin.value(),
            V_AC=self._vac_spin.value(),
            eps_r=self._eps_spin.value(),
            use_distributed_load=self._use_dist_load_cb.isChecked(),
            use_f0=self._use_f0_cb.isChecked(),
            bias_coupled_f0=self._bias_f0_cb.isChecked(),
            capacitance_gradient=self._cg_spin.value() * 1e-9,   # nF/m -> F/m
            lift_height=self._lift_spin.value() * 1e-9,          # nm -> m
        )

        method = self._method_combo.currentText()
        opts = SolverOptions(
            n_points=self._n_points_spin.value(),
            f_start=self._f_start_spin.value() * 1e3,
            f_stop=self._f_stop_spin.value() * 1e3,
            f_step=self._f_step_spin.value(),
            g=self._g_spin.value(),
            method=method,
        )

        return cant, contact, exc, opts

    def set_params(self, cant: CantileverParams, contact: ContactParams,
                   exc: ExcitationParams, opts: SolverOptions):
        """Load parameter dataclasses into the controls."""
        self._block_signals = True
        try:
            self._L_spin.setValue(cant.L * 1e6)
            self._w_spin.setValue(cant.w * 1e6)
            self._H_spin.setValue(cant.H * 1e6)
            self._k_spin.setValue(cant.k)
            self._phi_spin.setValue(math.degrees(cant.phi))
            self._m_spin.setValue(cant.m * 1e12)     # kg -> ng
            self._mT_spin.setValue(cant.mT * 1e12)
            self._tip_offset_spin.setValue(cant.tip_offset * 1e6)
            self._ow_ratio_spin.setValue(cant.overhang_width_ratio)

            self._k1_spin.setValue(contact.k1)
            self._k1_slider.setValue(_k1_to_slider(contact.k1))
            self._k2_spin.setValue(contact.k2)
            auto_k2 = abs(contact.k2 - 0.85 * contact.k1) < 1e-6 * contact.k1 + 1e-3
            self._k2_auto_cb.setChecked(auto_k2)
            self._k2_spin.setEnabled(not auto_k2)

            self._c1_spin.setValue(contact.c1 * 1e6)
            self._c2_spin.setValue(contact.c2 * 1e6)
            self._psi_spin.setValue(math.degrees(contact.psi))
            self._kcl_spin.setValue(_stiff_to_spin(contact.k_cone_lat))
            self._kca_spin.setValue(_stiff_to_spin(contact.k_cone_ax))

            # Mechanical drive
            amp_m = max(exc.d1, exc.d2)
            if amp_m == 0:
                amp_m = 1e-12
            amp_pm = amp_m * 1e12
            self._amp_spin.setValue(amp_pm)
            self._amp_slider.setValue(_amp_to_slider(amp_pm))

            if exc.d1 == 0 and exc.d2 == 0:
                self._drive_combo.setCurrentIndex(3)
            elif exc.d1 > 0 and exc.d2 == 0:
                self._drive_combo.setCurrentIndex(0)
            elif exc.d2 > 0 and exc.d1 == 0:
                self._drive_combo.setCurrentIndex(1)
            else:
                self._drive_combo.setCurrentIndex(2)

            # Electrostatics
            self._use_dist_load_cb.setChecked(exc.use_distributed_load)
            self._use_f0_cb.setChecked(exc.use_f0)
            self._vac_spin.setValue(exc.V_AC)
            self._dv_spin.setValue(exc.delta_V)
            self._eps_spin.setValue(exc.eps_r)
            self._lift_spin.setValue(exc.lift_height * 1e9)          # m -> nm
            self._f0_spin.setValue(abs(complex(exc.f0)) * 1e12)      # N -> pN
            self._cg_spin.setValue(exc.capacitance_gradient * 1e9)  # F/m -> nF/m
            self._bias_f0_cb.setChecked(exc.bias_coupled_f0)

            # Solver
            mid = self._method_combo.findText(opts.method)
            if mid >= 0:
                self._method_combo.setCurrentIndex(mid)
            self._f_start_spin.setValue(opts.f_start / 1e3)
            self._f_stop_spin.setValue(opts.f_stop / 1e3)
            self._f_step_spin.setValue(opts.f_step)
            self._n_points_spin.setValue(opts.n_points)
            self._g_spin.setValue(opts.g)
        finally:
            self._block_signals = False
        self.params_changed.emit()

    def save_to_json(self, path: str):
        """Serialize current parameters to a JSON file."""
        cant, contact, exc, opts = self.get_params()
        data = {
            "cantilever": {
                "L": cant.L, "w": cant.w, "H": cant.H,
                "k": cant.k, "m": cant.m, "mT": cant.mT,
                "phi": cant.phi, "tip_offset": cant.tip_offset,
                "overhang_width_ratio": cant.overhang_width_ratio,
            },
            "contact": {
                "k1": contact.k1, "k2": contact.k2,
                "c1": contact.c1, "c2": contact.c2,
                "psi": contact.psi,
                # JSON has no infinity; null means a rigid tip cone.
                "k_cone_lat": None if math.isinf(contact.k_cone_lat) else contact.k_cone_lat,
                "k_cone_ax": None if math.isinf(contact.k_cone_ax) else contact.k_cone_ax,
            },
            "excitation": {
                "d1": exc.d1, "d2": exc.d2,
                "f0": complex(exc.f0).real,
                "delta_V": exc.delta_V, "V_AC": exc.V_AC, "eps_r": exc.eps_r,
                "use_distributed_load": exc.use_distributed_load,
                "use_f0": exc.use_f0,
                "bias_coupled_f0": exc.bias_coupled_f0,
                "capacitance_gradient": exc.capacitance_gradient,
                "lift_height": exc.lift_height,
            },
            "solver": {
                "n_points": opts.n_points,
                "f_start": opts.f_start,
                "f_stop": opts.f_stop,
                "f_step": opts.f_step,
                "g": opts.g,
                "method": opts.method,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load_from_json(self, path: str):
        """Load parameters from a JSON file and populate the controls."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Missing keys fall back to the Multi 75-G defaults.
        d_cant, d_contact, d_exc, d_opts = default_params_multi75()

        c = data.get("cantilever", {})
        cant = CantileverParams(
            L=c.get("L", d_cant.L),
            w=c.get("w", d_cant.w),
            H=c.get("H", d_cant.H),
            k=c.get("k", d_cant.k),
            m=c.get("m", d_cant.m),
            mT=c.get("mT", d_cant.mT),
            phi=c.get("phi", d_cant.phi),
            tip_offset=c.get("tip_offset", d_cant.tip_offset),
            overhang_width_ratio=c.get("overhang_width_ratio",
                                       d_cant.overhang_width_ratio),
        )

        ct = data.get("contact", {})
        k_lat = ct.get("k_cone_lat")
        k_ax = ct.get("k_cone_ax")
        contact = ContactParams(
            k1=ct.get("k1", d_contact.k1),
            k2=ct.get("k2", d_contact.k2),
            c1=ct.get("c1", d_contact.c1),
            c2=ct.get("c2", d_contact.c2),
            psi=ct.get("psi", d_contact.psi),
            k_cone_lat=math.inf if k_lat is None else float(k_lat),
            k_cone_ax=math.inf if k_ax is None else float(k_ax),
        )

        ex = data.get("excitation", {})
        exc = ExcitationParams(
            d1=ex.get("d1", 1e-12),
            d2=ex.get("d2", 0.0),
            f0=ex.get("f0", 0.0),
            delta_V=ex.get("delta_V", 1.0),
            V_AC=ex.get("V_AC", 1.0),
            eps_r=ex.get("eps_r", 1.0),
            use_distributed_load=ex.get("use_distributed_load", False),
            use_f0=ex.get("use_f0", False),
            bias_coupled_f0=ex.get("bias_coupled_f0", False),
            capacitance_gradient=ex.get("capacitance_gradient", 50e-9),
            lift_height=ex.get("lift_height", 0.0),
        )

        s = data.get("solver", {})
        opts = SolverOptions(
            n_points=s.get("n_points", d_opts.n_points),
            f_start=s.get("f_start", d_opts.f_start),
            f_stop=s.get("f_stop", d_opts.f_stop),
            f_step=s.get("f_step", d_opts.f_step),
            g=s.get("g", d_opts.g),
            method=s.get("method", d_opts.method),
        )

        self.set_params(cant, contact, exc, opts)
