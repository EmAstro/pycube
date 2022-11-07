"""
Module to set matplotlib plot settings

.. include:: ../../docs/source/include/links.rst
"""

import matplotlib.pyplot as plt


def set_rc_params():
    """Set some rcParams for matplotlib to have uniform setting in all plots.

    This directly update rcParams with new values, so once is loaded the new setting will be there for the entire
    session
    """
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 28
    plt.rcParams["axes.titlesize"] = 30

    plt.rcParams['figure.figsize'] = 10, 10

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams["font.size"] = 30

    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.handletextpad"] = 1

    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams["lines.markeredgewidth"] = 3

    plt.rcParams["mathtext.fontset"] = 'dejavuserif'

    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['savefig.pad_inches'] = 0.03
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.format'] = 'pdf'

    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.direction"] = 'in'
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.labelsize"] = 22
    plt.rcParams["ytick.labelsize"] = 22
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1

    plt.rcParams["patch.linewidth"] = 5
    plt.rcParams["hatch.linewidth"] = 5



