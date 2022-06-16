# class set up for MUSE datacubes
import numpy as np
from pycube.core import background
from pycube import msgs
from astropy.io import fits

class Cube:
    def __init__(self, image,primary = None,data = None,stats = None, background_mode = None):
        self.image = image
        self.primary = primary
        self.data = data
        self.stats = stats

    def from_fits_file(self, fits_filename):
        hdul = fits.open(fits_filename)
        self.primary = hdul[0]
        self.data = hdul[1]
        self.stats = hdul[2]
        
"""
    def extract_vals(self):
        bgMedian = np.nanmedian(data_val,0)
        bgSigma = np.sqrt(np.nanmedian(stat_val,0))
        bgMask = np.zeros_like(data_val,0)
"""


    def background(self, mode='median'):
        if mode == 'median':
            backgrd = background.median_background(self.data)
        elif mode == 'sextractor':
            # ToDo implement this
            backgrd = background.sextractor_background(self.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))







        






