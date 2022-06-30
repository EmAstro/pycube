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
        # self._object = primary.header['OBJECT']

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
        self.hdul = fits.open(self.image)
        self.primary = self.hdul[0]
        self.data = self.hdul[1]
        self.stat = self.hdul[2]

    def get_background(self, min_lambda=None, max_lambda=None, maskZ=None, maskXY=None,
                       sigSourceDetection=5.0, minSourceArea=16., sizeSourceMask=6., maxSourceSize=50.,
                       maxSourceEll=0.9, edges=60, output='Object', debug=False,
                       showDebug=False):
        """
          Uses statBg from psf.py to generate the source mask and the background image with sources removed and appends
          to self.hdul for easy access
        Args:
            min_lambda:
            max_lambda:
            maskZ:
            maskXY:
            sigSourceDetection:
            minSourceArea:
            sizeSourceMask:
            maxSourceSize:
            maxSourceEll:
            edges:
            output:
            debug:
            showDebug:

        Returns:

        """

        datacopy = self.data.data
        statcopy = self.stat.data
        average_bg, median_bg, std_bg, var_bg,\
        pixels_bg, mask_bg_2D, bg_data_image = psf.statBg(dataCube=datacopy, statCube=statcopy,
                                                    min_lambda=min_lambda, max_lambda=max_lambda,
                                                    maskZ=maskZ, maskXY=maskXY,
                                                    sigSourceDetection=sigSourceDetection, minSourceArea=minSourceArea,
                                                    sizeSourceMask=sizeSourceMask, maxSourceSize=maxSourceSize,
                                                    maxSourceEll=maxSourceEll, edges=edges,
                                                    output=output, debug=debug, showDebug=showDebug)

        self.source_mask = fits.ImageHDU(data=mask_bg_2D, name='MASK')
        self.source_background = fits.ImageHDU(data=bg_data_image, name='BACKGROUND')
        self.hdul = self.hdul[:3] # removes MASK and BACKGROUND if function ran in succession
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
