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
import matplotlib.pyplot as plt


class IfuCube:
    def __init__(self, image, object=None, primary=None, data=None, stat=None, hdul=None, background_mode=None):
        """"
        Inputs:
            image: raw FITS file

        initializes data cube FITS file for IFU_cube class
        """
        self.image = image
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
        #self._object = primary.header['OBJECT']

    @property
    def source_mask(self):
        return self._source_mask

    @source_mask.setter
    def source_mask(self, source_mask):
        self._source_mask = source_mask


    def from_fits_file(self):
        """
        Opens .FITS file and separates information by primary, data, and stat.

        Inputs:
            fits_filename: .FITS file to assign self parameter to.
        Assigns:
            hdul to open data file
            Primary row of file
            Data row of file
            Stat (variance) row of file
        """
        # ToDo the value of those should be set by the spectrograph object
        # When the user specifies what spectrograph they utilize the code should tailor to assign these
        self.hdul = fits.open(self.image)
        self.primary = self.hdul[0]
        self.data = self.hdul[1]
        self.stat = self.hdul[2]

    def get_background(self,
                       sigSourceDetection=5.0, minSourceArea=16., sizeSourceMask=6., maxSourceSize=50.,
                       maxSourceEll=0.9, edges=60):
        """
          Uses statBg from psf.py to generate the source mask and the background
          image with sources removed and appends to self.hdul for easy access

        Inputs:
            min_lambda (int):
            min channel to create the image where to detect sources
        max_lambda (int):
            max channel to create the image where to detect sources
        maskZ
            when 1 (or True), this is a channel to be removed
        maskXY
            when 1 (or True), this spatial pixel will remove from
            the estimate of the b/g values
        sigSourceDetection (float):
            detection sigma threshold for sources in the
            collapsed cube. Defaults is 5.0
        minSourceArea (float):
            min area for source detection in the collapsed
            cube. Default is 16.
        sizeSourceMask (float):
            for each source, the model will be created in an elliptical
            aperture with size sizeSourceMask time the semi-minor and semi-major
            axis of the detection. Default is 6.
        maxSourceSize (float):
            sources with semi-major or semi-minor axes larger than this
            value will not be considered in the foreground source model.
            Default is 50.
        maxSourceEll (float):
            sources with ellipticity larger than this value will not be
            considered in the foreground source model. Default is 0.9.
        edges (int):
            frame size removed to avoid problems related to the edge
            of the image
        output (string):
            root file name for output
        Outputs:
            Attaches source mask and source background to hdul.

        """

        datacopy = self.data.data
        statcopy = self.stat.data
        datacopy = datacopy.byteswap(inplace=False).newbyteorder()
        statcopy = statcopy.byteswap(inplace=False).newbyteorder()
        cube_bg, mask_bg = psf.background_cube(datacube=datacopy, statcube=statcopy,
                                                    sigSourceDetection=sigSourceDetection, minSourceArea=minSourceArea,
                                                    sizeSourceMask=sizeSourceMask, maxSourceSize=maxSourceSize,
                                                    maxSourceEll=maxSourceEll, edges=edges)

        self.source_mask = fits.ImageHDU(data=mask_bg, name='MASK')
        self.source_background = fits.ImageHDU(data=cube_bg, name='BACKGROUND')
        self.hdul = self.hdul[:3] # removes MASK and BACKGROUND if function ran in succession
        self.hdul.append(self.source_mask)
        self.hdul.append(self.source_background)

    def background(self, mode='median'):
        if mode == 'median':
            self.background_mode = background.median_background(self.data.data)
        elif mode == 'sextractor':
            self.background_mode = background.sextractor_background(self.data.data, self.stat.data, )
        else:
            raise ValueError
            msgs.warning('Possible values are:\n {}'.format(background.BACKGROUND_MODES))
        embed()
