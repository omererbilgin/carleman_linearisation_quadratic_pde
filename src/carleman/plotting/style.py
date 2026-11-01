# academic article style type
from pathlib import Path


PLOT_WIDTH = 7.2


def plot_style(style=None):
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": .7,
        "lines.linewidth": 1.1,
        "text.usetex": False,
        **(style or {}),
    }


def save_figure(figure, filename, **kwargs):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, **kwargs)
