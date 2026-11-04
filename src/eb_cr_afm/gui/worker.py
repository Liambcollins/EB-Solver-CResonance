"""
Background computation workers for EB-CR-AFM GUI.

ComputationWorker: generic single-function background thread.
ProgressStiffnessWorker: runs compute_stiffness_frequency_map step-by-step
    (one k1 at a time) with live progress signals.
ProgressKappaWorker: runs kappa_map() with per-κ-point progress signals.
"""

from __future__ import annotations

import traceback
import numpy as np

from PySide6.QtCore import QThread, Signal


class ComputationWorker(QThread):
    """
    Generic background worker that calls fn(*args, **kwargs) in a thread.

    Signals
    -------
    result_ready(task_name, result)
        Emitted when the computation finishes successfully.
    error_occurred(task_name, traceback_str)
        Emitted when an exception is raised during computation.
    progress(int)
        Optional progress in [0, 100]; emitted when fn supports it via
        a ``progress_callback`` keyword argument.
    """

    result_ready = Signal(str, object)
    error_occurred = Signal(str, str)
    progress = Signal(int)

    def __init__(self, task_name: str, fn, args=(), kwargs=None, parent=None):
        super().__init__(parent)
        self._task_name = task_name
        self._fn = fn
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.result_ready.emit(self._task_name, result)
        except Exception:
            tb = traceback.format_exc()
            self.error_occurred.emit(self._task_name, tb)


class ProgressStiffnessWorker(QThread):
    """
    Runs compute_stiffness_frequency_map one k1 value at a time so that
    the GUI can show live progress.

    Emits the same signals as ComputationWorker.
    The result emitted via result_ready has the same schema as
    compute_stiffness_frequency_map:
        {'freqs': array(Nf), 'k1': array(Nk), 'amp_map': array(Nk,Nf),
         'phase_map': array(Nk,Nf)}
    """

    result_ready = Signal(str, object)
    error_occurred = Signal(str, str)
    progress = Signal(int)

    def __init__(self, cant, contact_base, exc, opts,
                 k1_values: np.ndarray,
                 k2_over_k1: float = 0.85,
                 obs_x: float = None,
                 parent=None):
        super().__init__(parent)
        self._cant = cant
        self._contact_base = contact_base
        self._exc = exc
        self._opts = opts
        self._k1_values = np.asarray(k1_values, dtype=float)
        self._k2_over_k1 = k2_over_k1
        self._obs_x = obs_x
        self._abort = False

    def abort(self):
        """Request early termination."""
        self._abort = True

    def run(self):
        try:
            from eb_cr_afm.params import ContactParams
            from eb_cr_afm.solver import frequency_sweep

            k1_values = self._k1_values
            freqs = self._opts.frequencies
            Nk = len(k1_values)
            Nf = len(freqs)

            amp_map_disp   = np.zeros((Nk, Nf))
            phase_map_disp = np.zeros((Nk, Nf))
            amp_map_slope  = np.zeros((Nk, Nf))
            phase_map_slope = np.zeros((Nk, Nf))

            for ik, k1 in enumerate(k1_values):
                if self._abort:
                    break

                mod_contact = ContactParams(
                    k1=k1,
                    k2=self._k2_over_k1 * k1,
                    c1=self._contact_base.c1,
                    c2=self._k2_over_k1 * self._contact_base.c1,
                    psi=self._contact_base.psi,
                )
                sweep = frequency_sweep(self._cant, mod_contact, self._exc, self._opts,
                                        obs_x=self._obs_x)
                amp_map_disp[ik, :]    = sweep["amplitudes_disp"]
                phase_map_disp[ik, :]  = sweep["phases_disp"]
                amp_map_slope[ik, :]   = sweep["amplitudes_slope"]
                phase_map_slope[ik, :] = sweep["phases_slope"]

                pct = int(100 * (ik + 1) / Nk)
                self.progress.emit(pct)

            result = {
                "freqs":            freqs,
                "k1":               k1_values,
                "amp_map_disp":     amp_map_disp,
                "phase_map_disp":   phase_map_disp,
                "amp_map_slope":    amp_map_slope,
                "phase_map_slope":  phase_map_slope,
                # backward compat
                "amp_map":   amp_map_disp,
                "phase_map": phase_map_disp,
            }
            self.result_ready.emit("stiffness_map", result)

        except Exception:
            tb = traceback.format_exc()
            self.error_occurred.emit("stiffness_map", tb)


class ProgressKappaWorker(QThread):
    """
    Runs kappa_map() in a background thread with per-κ-point progress signals.

    The κ grid is always dense around the Mode-1 zero crossing (κ≈28.4) and
    Mode-2 zero crossing (κ≈249), using the same concatenated linspace
    strategy as gen_kappa_map.py.

    Signals
    -------
    result_ready('kappa_map', dict)  — same schema as kappa_map()
    error_occurred('kappa_map', tb_str)
    progress(int)  — 0‥100
    """

    result_ready = Signal(str, object)
    error_occurred = Signal(str, str)
    progress = Signal(int)

    def __init__(self, cant, contact, opts,
                 kappa_max: float = 600.0,
                 n_kappa: int = 100,
                 n_modes: int = 2,
                 k2_over_k1: float = 0.0,
                 n_scan: int = 600,
                 n_refine: int = 200,
                 parent=None):
        super().__init__(parent)
        self._cant = cant
        self._contact = contact
        self._opts = opts
        self._kappa_max = float(kappa_max)
        self._n_kappa = int(n_kappa)
        self._n_modes = int(n_modes)
        self._k2_over_k1 = float(k2_over_k1)
        self._n_scan = int(n_scan)
        self._n_refine = int(n_refine)
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            from eb_cr_afm.spectra import kappa_map

            # Build a κ grid that is dense near the zero-crossings
            km = self._kappa_max
            n = self._n_kappa
            kappa_vals = np.unique(np.concatenate([
                np.linspace(0.0,   2.0,  max(5,  n // 20)),
                np.linspace(2.0,  60.0,  max(30, n // 3)),
                np.linspace(60.0, min(km, 300.0), max(30, n // 3)),
                np.linspace(min(km, 300.0), km,  max(10, n // 5)),
            ]))

            total = len(kappa_vals)

            def _progress(ik, _total):
                if not self._abort:
                    self.progress.emit(int(100 * (ik + 1) / _total))

            result = kappa_map(
                self._cant, self._contact, self._opts,
                kappa_values=kappa_vals,
                n_modes=self._n_modes,
                k2_over_k1=self._k2_over_k1,
                n_scan=self._n_scan,
                n_refine=self._n_refine,
                verbose=False,
                _progress_fn=_progress,
            )
            if not self._abort:
                self.result_ready.emit("kappa_map", result)

        except Exception:
            tb = traceback.format_exc()
            self.error_occurred.emit("kappa_map", tb)
