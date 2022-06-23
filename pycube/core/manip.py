import numpy as np
import sep
from photutils import EllipticalAperture

def find_sigma(array):
    """
    Simple expression to calculate Sigma quickly. Taking square root of median value.
    Inputs:
        array: 2D array of interest to generate sigma value
    Returns:
        Sigma value of given 2D array, ignoring NaNs
    """
    return np.sqrt(np.nanmedian(array))


def convert_to_wave(datacube, channels):
    wave = datacube.header['CRVAL3'] + (np.array(channels) * datacube.header['CD3_3'])
    return np.array(wave, dtype=float)


def collapse_cube(datacube, min_lambda = None, max_lambda = None):
    """
    Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    Inputs:
        datacube: 3D data file
        min_lambda: minimum wavelength
        max_lambda: maximum wavelength
    Returns:
        col_cube: Condensed 2D array of 3D file.
    """
    # safeguard -> if argument is Stat cube and is None
    if datacube is None:
        print("Object passed is None. Returning object..")
        return None
    datacopy = np.copy(datacube)
    z_max, y_max, x_max = np.shape(datacopy)
    # Checks and resets if outside boundaries of z
    if max_lambda > z_max or max_lambda is None:
        max_lambda = z_max
        print("Exceeded / unspecified wavelength in data cube. Max value is set to {}".format(int(z_max)))
    if min_lambda < 0 or min_lambda is None:
        min_lambda = 0
        print("Invalid / unspecified minimum wavelength. Min value is set to 0")

    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :])
    return col_cube

def elliptical_mask(datacube, x_position, y_position,
             semi_maj = None, semi_min = None,
             theta = 0, default = 10):
    """
    Inputs:
        datacube: 2D collapsed image (array)
        x_position: User given x coord of stellar object
        y_position: User given y coord of stellar object
        semi_maj: Semi-major axis. Set to default if not declared
        semi_min: Semi-minor axis. Set to 0.6 * default if not declared
        theta: angle for ellipse rotation around object, defaults to 0
        default: Pixel scale set to 10 if not specified
    Returns:
        Mask of 2D array of with user defined stellar objects
        denoted as 1 with all other elements 0
    """
    mask_array = np.zeros_like(datacube)
    object_position = [x_position,y_position]
    theta_rad = (theta * np.pi) / 180. #converts angle degrees to radians

    # if no position given..
    if x_position.size == 0:
        print("Missing [X] coordinate. No mask created.")
    elif y_position.size == 0:
        print("Missing [Y] coordinate. No mask created.")
    else:
        # results with default value. Left in place for testing
        if semi_maj is None:
            semi_maj = default
            print("Missing semi-major axis <- setting pixel value to {}".format(semi_maj))
        if semi_min is None:
            semi_min = default * 0.6
            print("Missing semi-minor axis <- setting pixel value to {}".format(semi_min))

        # creates ellipse around given coordinates and generates 2D mask of same shape as datacube
        object_ellipse = EllipticalAperture(object_position, semi_maj, semi_min, theta_rad)
        ellipse_mask = object_ellipse.to_mask().to_image(shape = np.shape(datacube))
        image_mask = mask_array + ellipse_mask

        return image_mask

def ra_dec_locate(datacube, ra, dec, theta = 0):
    """

    Inputs:
 # CRVAL1 * (xposition - CDELTA1)
 #
    Returns:

    """
    masked_array = np.zeros_like(datacube)
    theta_rad = (theta * np.pi) / 180. #converts angle degrees to radians
    for pixel in datacube:


# Debating implimenting
def elliptical_mask(datacube,
                        xObj,
                        yObj,
                        aObj=5.,
                        bObj=5.,
                        thetaObj=0.):
# 6/21 notes to self
# Multi object designation below.. could impliment as a list fed in and returned as a dictionary?
# assign array of object position with respective object # that could be assigned?
# if yes ^ impliment above in location function.

    for idxObj in range(0, len(xObj)):
        posObj = [xObj[idxObj], yObj[idxObj]]
    ellObj = EllipticalAperture(posObj, aObj[idxObj], bObj[idxObj], theta=thetaObj_rad[idxObj])
    ellMsk = ellObj.to_mask(method='center')[0].to_image(shape=imgData.shape)
    imgMsk = imgMsk + ellMsk

    imgMsk[imgMsk > 0.] = 1
    # Deleting temporary images to clear up memory
    if np.int(xObj.size) > 0:
        del posObj
    del ellObj
    del ellMsk

    return imgMsk.astype(int)


def source(datacube, ra, dec, z):
    """

    Args:
        datacube:
        ra:
        dec:
        z:

    Returns:

    """



