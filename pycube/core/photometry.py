import matplotlib.pyplot as plt
import numpy as np
import sep
import copy

from astropy.stats import sigma_clipped_stats
from astropy.io import ascii
from astropy.io import fits

from matplotlib import gridspec
from matplotlib.patches import Ellipse

from scipy import ndimage

from IPython import embed

from photutils import aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import EllipticalAperture
from photutils import centroids


def spectrum_from_circular_aperture(datacube, variancecube, x_aperture, y_aperture, radius_aperture=1.,
                                    maskcube=None, in_radius_background=None, out_radius_background=None):
    """Spectrum extraction over a circular aperture

    Args:
    datacube : np.array
        data in a 3D array
    variancecube : np.array
        variance in a 3D array
    x_aperture : float
        x-location of the source in pixel
    y_aperture : float
        y-location of the source in pixel
    radius_aperture : float
        radius where to perform the aperture photometry

    Returns
    -------
    fluxObj, errFluxObj
    """

    specApPhot = []
    specVarApPhot = []

    posObj  = [x_aperture, y_aperture]
    circObj = CircularAperture(posObj, r=radius_aperture)
    zMax, yMax, xMax = datacube.shape

    for channel in np.arange(0,zMax,1):
        # Total flux
        tempData = np.copy(datacube[channel,:,:])
        apPhot = aperture_photometry(tempData, circObj)
        # Error
        tempStat = np.copy(variancecube[channel,:,:])
        varApPhot = aperture_photometry(tempStat, circObj)
        # Loading lists
        specApPhot.append(apPhot['aperture_sum'][0])
        specVarApPhot.append(varApPhot['aperture_sum'][0])

    # Deleting temporary arrays to clear up memory
    del tempData
    del tempStat

    return np.array(specApPhot), np.power(np.array(specVarApPhot),0.5)
