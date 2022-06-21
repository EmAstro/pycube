import numpy as np
import sep

BACKGROUND_MODES = ['median', 'sextractor']


def median_background(datacube, sigma):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> default background taking median of 2D data

    Inputs:
        datacube: 2D array
        sigma: not implemented
    Returns:
        Median value of datacube background ignoring NaNs.
    """
    return np.nanmedian(datacube, 0)


def sextractor_background(datacube, sigma):
    """
    Backgrounds should only be implemented on 2D array.
    For 3D files, collapse to a desired dimension before passing function.
    -> Performs SEP function https://github.com/kbarbary/sep to set up background.

    Inputs:
        datacube: collapsed 2D array of 3D data
        sigma: 2D stat array converted in function
    Returns:
        SExtractor adjusted background of 2D array.
    """
    datacopy = np.copy(datacube)
    statcopy = np.copy(sigma)
    s_sigma = np.sqrt(np.nanmedian(statcopy))
    bg_median = np.nanmedian(datacopy)
    bg_mask = np.zeros_like(datacopy)

    bg_mask[(np.abs(datacopy - bg_median) > 7. * s_sigma)] = int(1)
    return sep.Background(datacopy, mask=bg_mask,
                          bw=64., bh=64.,
                          fw=5., fh=5.)