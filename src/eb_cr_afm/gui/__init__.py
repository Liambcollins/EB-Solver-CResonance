"""
CR2Beam GUI package (PySide6 + pyqtgraph).

Launch with ``cr2beam`` or ``python launch_gui.py``.  ``MainWindow`` is
imported lazily so that ``eb_cr_afm.gui.app`` can print a helpful message
when the optional GUI dependencies are missing.
"""

__all__ = ["MainWindow", "branding"]


def __getattr__(name):
    if name == "MainWindow":
        from .main_window import MainWindow
        return MainWindow
    if name == "branding":
        from . import branding
        return branding
    raise AttributeError(name)
