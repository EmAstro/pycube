import numpy as np
import sep
from pycube.core import manip
BACKGROUND_MODES = ['median', 'sextractor']


def median_background(datacontainer):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> default background taking median of 2D data

    Parameters
    ----------
    datacontainer : 2D array
        Collapsed data to calculate background from
    Returns
    -------
    float
        Median value of datacube's background ignoring NaNs.
    """
    return np.nanmedian(datacontainer)


def sextractor_background(datacontainer, statcube, var_value):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> Performs SEP function https://github.com/kbarbary/sep to set up background.

    Parameters
    ----------
    datacontainer : np.array
        Collapsed 2D array of 3D data
    statcube : np.array
        2D stat array converted in function
    var_value : int, float
        Affects the threshold parameter for normalizing the mask

    Returns
    -------
    SEP object
        SExtractor adjusted background of 2D array
    """
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = np.copy(datacontainer)
        statcopy = np.copy(statcube)

    s_sigma = manip.find_sigma(statcopy)
    bg_median = np.nanmedian(datacopy)
    bg_mask = np.zeros_like(datacopy)

    bg_mask[(np.abs(datacopy - bg_median) > var_value * s_sigma)] = 1
    return sep.Background(datacopy, mask=bg_mask,
                          bw=64., bh=64.,
                          fw=5., fh=5.)
