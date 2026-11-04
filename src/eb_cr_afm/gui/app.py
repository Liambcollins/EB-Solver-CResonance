"""Entry point for the CR2Beam desktop application.

Run it with ``cr2beam`` (after ``pip install -e ".[gui]"``) or
``python launch_gui.py`` from the repository root.
"""
import os
import sys


def main():
    # Tell pyqtgraph to use PySide6 (must be set before pyqtgraph is imported).
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.exit(
            "CR2Beam needs PySide6 and pyqtgraph. Install them with:\n"
            '    python -m pip install "PySide6>=6.5" "pyqtgraph>=0.13"'
        )

    from eb_cr_afm.gui import MainWindow, branding

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(branding.APP_NAME)
    app.setApplicationDisplayName(branding.APP_NAME)
    app.setOrganizationName(branding.ORG_NAME)
    app.setWindowIcon(branding.app_icon())
    app.setStyleSheet(branding.build_stylesheet())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
