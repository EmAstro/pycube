# class set up for MUSE datacubes
import numpy as np
from pycube.core import background
from pycube import msgs
from astropy.io import fits


class IFU_Cube:
    def __init__(self, image, primary=None, data=None, stat=None, background_mode=None):
        """
        initalizes data cube FITS file
        Args:
            image:
            primary:
            data:
            stat:
            background_mode:
        """
        self.image = image
        self.primary = primary
        self.data = data
        self.stat = stat

    def from_fits_file(self, fits_filename):
        hdul = fits.open(fits_filename)
        self.primary = hdul[0]
        self.data = hdul[1]
        self.stat = hdul[2]

    def extract_vals(self):
        bgMedian = np.nanmedian(self.data, 0)
        bgSigma = np.sqrt(np.nanmedian(self.stat, 0))
        bgMask = np.zeros_like(self.data, 0)

    def background(self, mode='median'):
        if mode == 'median':
            backgrd = background.median_background(self.data)
        elif mode == 'sextractor':
            backgrd = background.sextractor_background(self.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))
