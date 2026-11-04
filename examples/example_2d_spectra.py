"""Spatial-frequency map |z(x, f)| for the Bradler Table I cantilever.

Run from the repository root:  python examples/example_2d_spectra.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib
matplotlib.use("Agg")
from eb_cr_afm import default_params, compute_2d_spectra, plot_2d_spectra, SolverOptions

cant, contact, exc, opts = default_params()
opts_2d = SolverOptions(n_points=150, f_start=300e3, f_stop=400e3, f_step=1000.0, g=opts.g)
print("Computing 2D spectra...")
spectra = compute_2d_spectra(cant, contact, exc, opts_2d, n_x_points=60, verbose=True)

out_dir = os.path.join(HERE, "..", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "2d_spectra.png")
plot_2d_spectra(spectra, save_path=out)
print("Saved", os.path.abspath(out))
