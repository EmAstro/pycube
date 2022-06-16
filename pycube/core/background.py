import numpy as np
BACKGROUND_MODES = ['median', 'sextractor']

def median_background(datacube, sigma):
    return np.nanmedian(datacube, 0)

def sextractor_background(datacube, sigma):
    # reading documentation TODO
    return None