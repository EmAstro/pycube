# class set up for MUSE datacubes
"""Import modules useful for analyzing MUSE data and handling FITS files"""
import numpy as np

from pycube.core import background
from pycube import psf
from pycube import msgs as msgs

from astropy.io import fits


class IfuCube:
    def __init__(self, image, instrument=None, object=None,
                 primary=None, data=None, stat=None, hdul=None,
                 background_mode=None):
        """"
        Inputs:
            image: raw FITS file

        initializes data cube FITS file for IFU_cube class
        """
        self.image = image
        self.instrument = instrument
        self.object = object
        self.primary = primary
        self.data = data
        self.stat = stat
        self.source_mask = None
        self.source_background = None
        self.hdul = hdul
        self.background_mode = background_mode

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, primary):
        self._primary = primary
        # self._object = primary.header['OBJECT']

    @property
    def source_mask(self):
        return self._source_mask

    @source_mask.setter
    def source_mask(self, source_mask):
        self._source_mask = source_mask

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    def initialize_file(self):
        """
        Opens file and separates information by primary, data, and stat.

        Assigns
        -------
        hdul :
            full data table
        primary :
            primary row of hdul
        data :
            data row of hdul
        stat :
            stat (variance) row of hdul
        """

        self.hdul = fits.open(self.image, memmap=True)
        self.primary = self.hdul[self.instrument.primary_extension]
        self.data = self.hdul[self.instrument.data_extension]
        self.stat = self.hdul[self.instrument.sigma_extension]

    # functions used for pulling IFUcube information within module functions
    def get_primary(self):
        return self.primary.header

    def get_data(self):
        return np.copy(self.data.data)

    def get_data_header(self):
        return self.data.header

    def get_stat(self):
        return np.copy(self.stat.data)

    def get_stat_header(self):
        return self.stat.header

    def get_data_stat(self):
        return self.get_data(), self.get_stat()

    def get_headers(self):
        return self.get_data_header(), self.get_stat_header()

    def get_dimensions(self):
        z_max, y_max, x_max = np.shape(self.get_data())
        return z_max, y_max, x_max

    def get_background(self,
                       sig_source_detection=5.0, min_source_area=16.,
                       source_mask_size=6., edges=60):
        """Uses statBg from psf.py to generate the source mask and the background
        image with sources removed and appends to self.hdul for easy access

        Parameters
        ----------
        sig_source_detection : float
            detection sigma threshold for sources in the
            collapsed cube. Defaults is 5.0
        min_source_area : float
            min area for source detection in the collapsed
            cube. Default is 16.
        source_mask_size : float
            for each source, the model will be created in an elliptical
            aperture with size source_mask_size time the semi-minor and semi-major
            axis of the detection (default is 6.)
        edges : int
            frame size removed to avoid problems related to the edge
            of the image

        Returns
        -------
        astropy.hdul
            Attaches source mask and source background to hdul

        """

        cube_bg, mask_bg = psf.background_cube(self, sig_source_detect=sig_source_detection,
                                               min_source_area=min_source_area,
                                               source_mask_size=source_mask_size,
                                               edges=edges)

        self.source_mask = fits.ImageHDU(data=mask_bg, name='MASK')
        self.source_background = fits.ImageHDU(data=cube_bg, name='BACKGROUND')
        self.hdul = self.hdul[:3]  # removes MASK and BACKGROUND if function ran in succession
        self.hdul.append(self.source_mask)
        self.hdul.append(self.source_background)


    def background(self, mode='median'):
        if mode == 'median':
            self.background_mode = background.median_background(self.data.data)
        elif mode == 'sextractor':
            self.background_mode = background.sextractor_background(self.data.data, self.stat.data)
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))
