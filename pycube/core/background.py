import numpy as np
from scipy import stats

BACKGROUND_MODES = ['median', 'sextractor']


def median_background(datacube, sigma):
    return np.nanmedian(datacube, 0)

def sextractor_background(datacube, sigma):
    return stats.sigmaclip(datacube, high=sigma)
