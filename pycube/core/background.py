import numpy as np
import sep
from pycube.core import manip
BACKGROUND_MODES = ['median', 'sextractor']


def median_background(datacube, sigma):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> default background taking median of 2D data

    Inputs:
        datacontainers: 2D array
        sigma: not implemented
    Returns:
        Median value of datacontainers background ignoring NaNs.
    """
    return np.nanmedian(datacube, 0)


def sextractor_background(datacube, statcube, var_value=5.):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> Performs SEP function https://github.com/kbarbary/sep to set up background.

    Inputs:
        datacontainers(array): collapsed 2D array of 3D data
        sigma(array): 2D stat array converted in function
        var_value(int or float): affects the threshold parameter for normalizing the mask
    Returns:
        SExtractor adjusted background of 2D array.(sep object)
    """
    datacopy = np.copy(datacube)
    statcopy = np.copy(statcube)
    s_sigma = manip.find_sigma(statcopy)
    bg_median = np.nanmedian(datacopy)
    bg_mask = np.zeros_like(datacopy)

    bg_mask[(np.abs(datacopy - bg_median) > var_value * s_sigma)] = int(1)
    return sep.Background(datacopy, mask=bg_mask,
                          bw=64., bh=64.,
                          fw=5., fh=5.)
