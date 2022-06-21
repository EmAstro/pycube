# class set up for MUSE datacubes
"""Import modules useful for analyzing MUSE data and handling FITS files"""
import numpy as np
from pycube.core import background
from pycube.core import manip
from pycube import msgs
from astropy.io import fits
from IPython import embed
import sep
from matplotlib import patches


class IFU_cube:
    def __init__(self, image, primary=None, data=None, stat=None, background_mode=None):
        """"
        Inputs:
            image: raw FITS file

        initializes data cube FITS file for IFU_cube class
        """
        self.image = image
        self.primary = primary
        self.data = data
        self.stat = stat

    def from_fits_file(self, fits_filename):
        """
        Opens .FITS file and separates information by primary, data, and stat.
        Inputs:
            fits_filename: .FITS file to assign self parameter to.
        Assigns:
            Primary, data, stat, and dimensions of array.
        """
        hdul = fits.open(fits_filename)
        self.primary = hdul[0]
        self.data = hdul[1]
        self.stat = hdul[2]
        self.z_max, self.y_max, self.x_max = np.shape(self.data.data)

    def extract_vals(self):
        bg_sigma = np.sqrt(np.nanmedian(self.stat.data))
        bg_variance = np.nanvar(self.data.data)
        bg_average = np.nanmean(self.stat.data)
        bg_scale_cor = bg_variance / bg_average
        bg_mask = np.zeros_like(self.data.data)
        self.channels = np.arange(0, self.z_max, 1, dtype=int)

    def convert_to_wave(self, channels):
        wave = self.data.header['CRVAL3'] + (np.array(channels) * self.data.header['CD3_3'])
        self.wave_cube = np.array(wave, dtype=float)

    def background(self, mode='median'):
        if mode == 'median':
            backgrd = background.median_background(self.data.data)
        elif mode == 'sextractor':
            backgrd = background.sextractor_background(self.data.data,self.stat.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))


    def position(self, ra, dec, z):
        ra = self.primary.header['RA']
        dec = self.primary.header['DEC']
