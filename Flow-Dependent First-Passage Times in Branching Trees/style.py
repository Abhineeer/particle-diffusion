"""
Shared figure settings so every figure matches.
Physical Review widths: single column 8.5 cm, full page 17.8 cm.
"""

import matplotlib.pyplot as plt

SINGLE = 3.375 # inches, 8.5 cm
DOUBLE = 7.0 # inches, 17.8 cm

# one fixed colour per branch count, used in every figure
B_COLOURS = {1: "#1b1b1b", 2: "#c0392b", 3: "#2471a3", 4: "#7d3c98",
             5: "#1e8449", 6: "#b9770e"}


def setup():
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"