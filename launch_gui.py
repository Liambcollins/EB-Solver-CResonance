#!/usr/bin/env python3
"""Launch the CR2Beam GUI from the repository root: ``python launch_gui.py``.

If you installed the package with ``pip install -e ".[gui]"`` you can also
just type ``cr2beam``.
"""
import os
import sys

# Make the package importable even if it has not been pip-installed.
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from eb_cr_afm.gui.app import main

if __name__ == "__main__":
    main()
