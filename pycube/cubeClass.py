# class set up for MUSE datacubes
"""Import modules useful for analyzing MUSE data and handling FITS files"""
import numpy as np
from pycube.core import background
from pycube.core import manip
from pycube import psf
from pycube import msgs
from astropy.io import fits
from IPython import embed
import sep


class IfuCube:
    def __init__(self, image, object=None, primary=None, data=None, stat=None, background_mode=None):
        """"
        Inputs:
            image: raw FITS file

        initializes data cube FITS file for IFU_cube class
        """
        self.image = image
        self.primary = primary
        self.object = object
        self.data = data
        self.stat = stat
        self.source_mask = None
        self.background = None

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, primary):
        self._primary = primary
        self._object = primary.header['OBJECT']

    def from_fits_file(self):
        """
        Opens .FITS file and separates information by primary, data, and stat.
        Inputs:
            fits_filename: .FITS file to assign self parameter to.
        Assigns:
            Primary, data, stat, and dimensions of array.
        """
        hdul = fits.open(self.image)
        self.primary = hdul[0]
        self.data = hdul[1]
        self.stat = hdul[2]
        # Ema: this will not work because the attributes are not set in the __init__
        self.z_max, self.y_max, self.x_max = np.shape(self.data.data)

    def get_background(self, mode, min_lambda, max_lambda):

        datacopy = self.data.data
        statcopy = self.stat.data
        stat_2_d = manip.collapse_cube(statcopy, min_lambda, max_lambda)
        data_2_d = manip.collapse_cube(datacopy, min_lambda, max_lambda)
        x_pos, y_pos, semi_maj, semi_min, theta, all_objects = psf.find_sources(data_2_d, stat_2_d, min_lambda, max_lambda)
        void_mask = np.zeros_like(data_2_d)
        source_mask = manip.location(data_2_d, x_pos, y_pos, semi_min, semi_maj, theta)








        return(image_mask, background_cube)

    def extract_vals(self):
        bg_sigma = np.sqrt(np.nanmedian(self.stat.data))
        bg_variance = np.nanvar(self.data.data)
        bg_average = np.nanmean(self.stat.data)
        bg_scale_cor = bg_variance / bg_average
        bg_mask = np.zeros_like(self.data.data)
        self.channels = np.arange(0, self.z_max, 1, dtype=int)
        print(bg_sigma, bg_variance, bg_average, bg_scale_cor, bg_mask)


    def background(self, mode='median'):
        if mode == 'median':
            backgrd = background.median_background(self.data.data)
        elif mode == 'sextractor':
            backgrd = background.sextractor_background(self.data.data,self.stat.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))

