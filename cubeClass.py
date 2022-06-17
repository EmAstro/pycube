# class set up for MUSE datacubes
"""Import modules useful for analyzing MUSE data and handling FITS files"""
import numpy as np
from pycube.core import background
from pycube import msgs
from astropy.io import fits
from IPython import embed
import sep

class IFU_cube:
    def __init__(self, image, primary=None, data=None, stat=None, background_mode=None):
        """
        initalizes data cube FITS file
        Args:
            image: FITS image file
            primary: primary header/data in from_fits_file function
            data: data header/data in from_fits_file function
            stat: stat header/data in from_fits_file function
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
        bgSigma = np.sqrt(np.nanmedian(self.stat, 0))

    def background(self, mode='median'):
        if mode == 'median':
            backgrd = background.median_background(self.data)
        elif mode == 'sextractor':
            backgrd = background.sextractor_background(self.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))
        bg_mask = np.zeros_like(self.data, 0)
        img_backgrd = sep.backgrd(self.data,
                                   mask = bg_mask,
                                   bw = 64.,bh = 64.,
                                   fw = 5., fh = 5.)

    def position(self,ra,dec,z):



