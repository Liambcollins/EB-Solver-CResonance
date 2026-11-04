"""Reproduce the Table I transfer function of Bradler et al., J. Appl. Phys. 120, 165107 (2016).

Run from the repository root:  python examples/example_table1_simulation.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib
matplotlib.use("Agg")
from eb_cr_afm import default_params, frequency_sweep, sho_analysis, plot_transfer_function

cant, contact, exc, opts = default_params()          # Table I parameters
sweep = frequency_sweep(cant, contact, exc, opts, verbose=True)
d = max(sho_analysis(sweep), key=lambda m: m["A_res"])
print(f"f_res = {d['f_res']/1e3:.2f} kHz")
print(f"|z'|  = {d['A_res']*1e6:.3f} urad at the peak")
print(f"Q     = {d['Q']:.0f}")

out_dir = os.path.join(HERE, "..", "results")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "table1_transfer_function.png")
plot_transfer_function(sweep, save_path=out)
print("Saved", os.path.abspath(out))
