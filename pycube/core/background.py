import numpy as np


BACKGROUND_MODES = ['median', 'sextractor']


def median_background(datacube, sigma):
    return np.nanmedian(datacube, 0)

def sextractor_background(datacube, sigma):
    return (2.5 * np.nanmedian(datacube)) - (1.5 * np.nanmean(datacube))
