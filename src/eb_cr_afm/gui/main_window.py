"""
Main window for the EB-CR-AFM interactive GUI.

Layout: QSplitter(horizontal) with left=ParameterPanel, right=QTabWidget.
Tabs: Transfer Function, Mode Shape, 2D Map.
"""

from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QTabWidget,
    QStatusBar, QLabel, QFileDialog,
    QMessageBox, QApplication, QWidget, QVBoxLayout,
)
from PySide6.QtGui import QAction

from .parameter_panel import ParameterPanel
from .views.transfer_function import TransferFunctionView
from .views.mode_shape import ModeShapeView
from .views.spectra_map import SpectraMapView
from .worker import ComputationWorker, ProgressStiffnessWorker, ProgressKappaWorker
from . import branding

from eb_cr_afm.solver import frequency_sweep, solve_single_frequency
from eb_cr_afm.spectra import compute_2d_spectra

# Configure pyqtgraph globally: clean light theme for a publication look.
pg.setConfigOptions(antialias=True, background="w", foreground=branding.Palette.INK)


class MainWindow(QMainWindow):
    """
    Top-level application window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{branding.APP_NAME} — {branding.APP_TAGLINE}")
        self.setWindowIcon(branding.app_icon())
        self.resize(1400, 880)

        # Active worker references (current job for each task type)
        self._sweep_worker: Optional[ComputationWorker] = None
        self._mode_worker: Optional[ComputationWorker] = None
        self._map_worker: Optional[ProgressStiffnessWorker] = None
        self._spatial_worker: Optional[ComputationWorker] = None
        self._kappa_worker: Optional[ProgressKappaWorker] = None

        # Graveyard: keeps Python refs alive until C++ thread finishes,
        # preventing "QThread destroyed while running" crashes.
        self._live_workers: list = []

        # Cached results
        self._last_sweep: Optional[dict] = None
        self._last_sho: Optional[list] = None

        # Debounce timer for param changes -> auto-recompute sweep
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(600)
        self._debounce_timer.timeout.connect(self._run_transfer_function)

        self._build_ui()
        self._build_menu()
        self._build_status_bar()

        # Trigger initial computation after event loop starts
        QTimer.singleShot(200, self._run_transfer_function)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)

        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Branded header strip
        outer.addWidget(branding.HeaderBar())

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: parameter panel
        self._param_panel = ParameterPanel()
        self._param_panel.setMinimumWidth(360)
        self._param_panel.setMaximumWidth(460)
        splitter.addWidget(self._param_panel)

        # Right: tab widget
        self._tabs = QTabWidget()
        self._tf_view = TransferFunctionView()
        self._ms_view = ModeShapeView()
        self._map_view = SpectraMapView()

        self._tabs.addTab(self._tf_view, "Transfer Function")
        self._tabs.addTab(self._ms_view, "Mode Shape")
        self._tabs.addTab(self._map_view, "2D Map")
        splitter.addWidget(self._tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 960])

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 4)
        body_layout.addWidget(splitter)
        outer.addWidget(body, 1)

        # Wire signals
        self._param_panel.params_changed.connect(self._on_params_changed)
        self._param_panel.compute_tf_requested.connect(self._run_transfer_function)
        self._param_panel.compute_map_requested.connect(self._on_map_requested)

        self._tf_view.frequency_selected.connect(self._on_frequency_selected)
        self._tf_view.obs_settings_changed.connect(self._run_transfer_function)
        self._tf_view.sho_updated.connect(self._on_sho_updated)
        self._ms_view.solve_requested.connect(self._run_mode_shape)

        self._map_view.stiffness_map_requested.connect(self._on_stiffness_map_requested)
        self._map_view.spatial_map_requested.connect(self._on_spatial_map_requested)
        self._map_view.kappa_map_requested.connect(self._on_kappa_map_requested)

    def _build_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        act_load = QAction("Load Params (JSON)...", self)
        act_load.triggered.connect(self._action_load_params)
        file_menu.addAction(act_load)

        act_save = QAction("Save Params (JSON)...", self)
        act_save.triggered.connect(self._action_save_params)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_export_fig = QAction("Export Figure (PNG/SVG/PDF)...", self)
        act_export_fig.triggered.connect(self._action_export_figure)
        file_menu.addAction(act_export_fig)

        act_export_data = QAction("Export Data (NPZ)...", self)
        act_export_data.triggered.connect(self._action_export_data)
        file_menu.addAction(act_export_data)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(QApplication.quit)
        file_menu.addAction(act_quit)

        # View menu
        view_menu = menubar.addMenu("View")
        act_reset = QAction("Reset Zoom", self)
        act_reset.triggered.connect(self._action_reset_zoom)
        view_menu.addAction(act_reset)

        # Help menu
        help_menu = menubar.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._action_about)
        help_menu.addAction(act_about)

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._status_label = QLabel("Ready")
        self._res_label = QLabel("")
        self._res_label.setObjectName("ResLabel")
        self._res_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        sb.addWidget(self._status_label, 1)
        sb.addPermanentWidget(self._res_label)

    # ------------------------------------------------------------------
    # Param change debounce
    # ------------------------------------------------------------------

    def _on_params_changed(self):
        self._debounce_timer.start()

    # ------------------------------------------------------------------
    # Worker lifetime helpers
    # ------------------------------------------------------------------

    def _retire_worker(self, worker):
        """Disconnect signals from a running worker and park it in the
        graveyard so Python keeps a reference until the C++ thread exits."""
        if worker is None:
            return
        for sig in (worker.result_ready, worker.error_occurred, worker.progress):
            with warnings.catch_warnings():
                # PySide6 warns when disconnecting a signal with no slots.
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if worker.isRunning():
            # Ask thread's event loop to quit (won't kill the run() body,
            # but keeps the QThread alive until run() returns naturally).
            worker.abort() if hasattr(worker, "abort") else None
            self._live_workers.append(worker)
            worker.finished.connect(lambda w=worker: self._live_workers.remove(w)
                                    if w in self._live_workers else None)

    # ------------------------------------------------------------------
    # Transfer function computation
    # ------------------------------------------------------------------

    def _run_transfer_function(self):
        self._retire_worker(self._sweep_worker)

        cant, contact, exc, opts = self._param_panel.get_params()
        obs_from_tip = self._tf_view.get_obs_from_tip_m()
        obs_x = cant.L - obs_from_tip  # measured from physical tip end
        self._status_label.setText("Computing transfer function...")

        self._sweep_worker = ComputationWorker(
            "sweep", frequency_sweep,
            args=(cant, contact, exc, opts, obs_x),
        )
        self._sweep_worker.result_ready.connect(self._on_worker_result)
        self._sweep_worker.error_occurred.connect(self._on_worker_error)
        self._sweep_worker.start()

    def _on_sweep_done(self, result: dict):
        self._last_sweep = result
        # The view fits the peaks on whichever observable (displacement or
        # slope) is displayed and reports them back through sho_updated.
        self._tf_view.update_sweep(result)
        self._status_label.setText("Transfer function ready.")

    def _on_sho_updated(self, sho_modes):
        """Show the strongest fitted contact resonance in the status bar."""
        self._last_sho = sho_modes
        if sho_modes:
            dominant = max(sho_modes, key=lambda m: m["A_res"])
            self._res_label.setText(
                f"f_CR = {dominant['f_res']/1e3:.2f} kHz  "
                f"Q = {dominant['Q']:.0f}"
            )
        else:
            self._res_label.setText("")

    # ------------------------------------------------------------------
    # Mode shape computation
    # ------------------------------------------------------------------

    def _on_frequency_selected(self, freq_hz: float):
        self._ms_view.set_frequency(freq_hz)
        self._run_mode_shape(freq_hz)

    def _run_mode_shape(self, freq_hz: float):
        self._retire_worker(self._mode_worker)

        cant, contact, exc, opts = self._param_panel.get_params()
        self._status_label.setText(f"Solving mode shape at {freq_hz/1e3:.2f} kHz...")

        self._mode_worker = ComputationWorker(
            "mode", solve_single_frequency,
            args=(freq_hz, cant, contact, exc, opts),
        )
        self._mode_worker.result_ready.connect(self._on_worker_result)
        self._mode_worker.error_occurred.connect(self._on_worker_error)
        self._mode_worker.start()

    def _on_mode_done(self, result: dict):
        self._ms_view.update_profile(result)
        freq_khz = result["freq"] / 1e3
        self._status_label.setText(f"Mode shape at {freq_khz:.2f} kHz ready.")

    # ------------------------------------------------------------------
    # 2D map computation
    # ------------------------------------------------------------------

    def _on_map_requested(self):
        """Triggered from 'Compute 2D Map' button in parameter panel."""
        self._tabs.setCurrentIndex(2)
        k1_min, k1_max, n_k1 = self._map_view.get_k1_range()
        obs_from_tip_m, _ = self._map_view.get_obs_settings()
        self._run_stiffness_map(k1_min, k1_max, n_k1, obs_from_tip_m)

    def _on_stiffness_map_requested(self, k1_min: float, k1_max: float, n_k1: int,
                                    obs_from_tip_m: float):
        self._run_stiffness_map(k1_min, k1_max, n_k1, obs_from_tip_m)

    def _on_spatial_map_requested(self, n_x: int):
        self._run_spatial_map(n_x)

    def _run_stiffness_map(self, k1_min: float, k1_max: float, n_k1: int,
                           obs_from_tip_m: float = 14e-6):
        self._retire_worker(self._map_worker)

        cant, contact, exc, opts = self._param_panel.get_params()
        obs_x = cant.L - obs_from_tip_m  # measured from physical tip end
        k1_values = np.logspace(np.log10(k1_min), np.log10(k1_max), n_k1)

        self._status_label.setText("Computing stiffness map...")
        self._map_view.show_progress(0)

        self._map_worker = ProgressStiffnessWorker(
            cant, contact, exc, opts, k1_values,
            obs_x=obs_x,
        )
        self._map_worker.result_ready.connect(self._on_worker_result)
        self._map_worker.error_occurred.connect(self._on_worker_error)
        self._map_worker.progress.connect(self._map_view.show_progress)
        self._map_worker.start()

    def _run_spatial_map(self, n_x: int):
        self._retire_worker(self._spatial_worker)

        cant, contact, exc, opts = self._param_panel.get_params()
        self._status_label.setText("Computing spatial map...")
        self._map_view.show_progress(0)

        self._spatial_worker = ComputationWorker(
            "spatial_map", compute_2d_spectra,
            args=(cant, contact, exc, opts, n_x),
        )
        self._spatial_worker.result_ready.connect(self._on_worker_result)
        self._spatial_worker.error_occurred.connect(self._on_worker_error)
        self._spatial_worker.start()

    def _on_map_done(self, result: dict):
        self._map_view.update_stiffness_map(result)
        self._map_view.show_progress(100)
        self._status_label.setText("Stiffness map ready.")

    def _on_spatial_map_done(self, result: dict):
        self._map_view.update_spatial_map(result)
        self._map_view.show_progress(100)
        self._status_label.setText("Spatial map ready.")

    def _on_kappa_map_requested(self, kappa_max: float, n_kappa: int, n_modes: int):
        self._run_kappa_map(kappa_max, n_kappa, n_modes)

    def _run_kappa_map(self, kappa_max: float, n_kappa: int, n_modes: int):
        self._retire_worker(self._kappa_worker)

        cant, contact, _exc, opts = self._param_panel.get_params()
        self._status_label.setText("Computing κ-map...")
        self._map_view.show_progress(0)

        self._kappa_worker = ProgressKappaWorker(
            cant, contact, opts,
            kappa_max=kappa_max,
            n_kappa=n_kappa,
            n_modes=n_modes,
        )
        self._kappa_worker.result_ready.connect(self._on_worker_result)
        self._kappa_worker.error_occurred.connect(self._on_worker_error)
        self._kappa_worker.progress.connect(self._map_view.show_progress)
        self._kappa_worker.start()

    def _on_kappa_map_done(self, result: dict):
        self._map_view.update_kappa_map(result)
        self._map_view.show_progress(100)
        self._status_label.setText("κ-map ready.")

    # ------------------------------------------------------------------
    # Generic worker result dispatcher
    # ------------------------------------------------------------------

    def _on_worker_result(self, task_name: str, result):
        if task_name == "sweep":
            self._on_sweep_done(result)
        elif task_name == "mode":
            self._on_mode_done(result)
        elif task_name == "stiffness_map":
            self._on_map_done(result)
        elif task_name == "spatial_map":
            self._on_spatial_map_done(result)
        elif task_name == "kappa_map":
            self._on_kappa_map_done(result)

    def _on_worker_error(self, task_name: str, tb_str: str):
        self._status_label.setText(f"Error in '{task_name}'. See console.")
        print(f"[EB-CR-AFM GUI] Error in worker '{task_name}':\n{tb_str}")
        QMessageBox.critical(
            self, "Computation Error",
            f"Task '{task_name}' failed:\n\n{tb_str[:800]}"
        )

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------

    def _action_load_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "",
            "JSON files (*.json);;All files (*)"
        )
        if path:
            try:
                self._param_panel.load_from_json(path)
                self._run_transfer_function()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def _action_save_params(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "params.json",
            "JSON files (*.json);;All files (*)"
        )
        if path:
            try:
                self._param_panel.save_to_json(path)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _action_export_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "figure.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All files (*)"
        )
        if path:
            self._export_figure(path)

    def _action_export_data(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "data.npz",
            "NumPy archive (*.npz);;All files (*)"
        )
        if path:
            self._export_data(path)

    def _action_reset_zoom(self):
        idx = self._tabs.currentIndex()
        if idx == 0:
            self._tf_view._auto_range()
        elif idx == 1:
            self._ms_view._z_plot.autoRange()
            self._ms_view._zp_plot.autoRange()
        elif idx == 2:
            map_mode = self._map_view._mode_combo.currentIndex()
            if map_mode == 2:
                self._map_view._kf_plot.autoRange()
                self._map_view._kzp_plot.autoRange()
                self._map_view._kzl_plot.autoRange()
            else:
                self._map_view._amp_plot.autoRange()
                self._map_view._phase_plot.autoRange()

    def _action_about(self):
        QMessageBox.about(
            self, f"About {branding.APP_NAME}",
            f"<b>{branding.APP_NAME}</b> v{branding.APP_VERSION}<br>"
            f"{branding.APP_TAGLINE}<br><br>"
            f"{branding.ORG_NAME}<br><br>"
            "Interactive explorer for the Euler–Bernoulli contact-resonance "
            "AFM / PFM solver, including the tilted two-segment overhang model "
            "for near-tip displacement nulls.<br><br>"
            "Based on Bradler <i>et al.</i>, J. Appl. Phys. 120, 165107 (2016).<br>"
            "Built with PySide6 + pyqtgraph."
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_figure(self, path: str):
        """Save the current tab's pyqtgraph widget as an image."""
        idx = self._tabs.currentIndex()
        try:
            import pyqtgraph.exporters as exporters
            if idx == 0:
                item = self._tf_view._amp_plot.plotItem
            elif idx == 1:
                item = self._ms_view._z_plot.plotItem
            else:
                item = self._map_view._amp_plot

            suffix = Path(path).suffix.lower()
            if suffix in (".svg",):
                exp = exporters.SVGExporter(item)
            else:
                exp = exporters.ImageExporter(item)
            exp.export(path)
            self._status_label.setText(f"Figure saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_data(self, path: str):
        """Save current sweep data as NPZ with a JSON params sidecar."""
        if self._last_sweep is None:
            QMessageBox.information(self, "No data", "Run a sweep first.")
            return
        try:
            sweep = self._last_sweep
            np.savez_compressed(
                path,
                freqs=sweep["freqs"],
                amplitudes=sweep["amplitudes"],
                phases=sweep["phases"],
            )
            # Params sidecar
            params_path = str(Path(path).with_suffix(".params.json"))
            self._param_panel.save_to_json(params_path)
            self._status_label.setText(f"Data saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
