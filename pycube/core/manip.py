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


def collapse_cube(datacube, min_lambda, max_lambda):
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
    if max_lambda > z_max:
        max_lambda = z_max
        print("Exceed wavelength in datacube. Max value is set to {}".format(int(z_max)))
    if min_lambda < 0:
        min_lambda = 0
        print("Invalid wavelength value for min. Min value is set to 0")

    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :])
    return col_cube


def location(datacube, x_position, y_position,
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


















# Debating implimenting
def elliptical_mask(datacube,
                        xObj,
                        yObj,
                        aObj=5.,
                        bObj=5.,
                        thetaObj=0.):
    """Returning a mask where sources are marked with 1 and background
    with 0. It is just the superposition of several elliptical masks
    centered at xObj and yObj with axis aObj and bObj.
    Parameters
    ----------
    imgData : np.array
        data in a 2D array. The mask will have the same size
        of this.
    xObj : np.array
        x-location of the sources in pixels
    yObj : np.array
        y-location of the sources in pixels
    aObj
        semi-major axis in pixel (i.e. the radius if
        aObj=bObj)
    bObj
        semi-minor axis in pixel (i.e. the radius if
        aObj=bObj)
    thetaObj
        angle wrt the x-axis in degrees
    Returns
    -------
    imgMsk : np.array
        mask where sources are marked with 1 and background with 0.
        It has the same dimensions of the input imgData
    """


    # converting degrees to radians
    # ToDo: double check that this is the correct input for EllipticalAperture
    thetaObj_rad = thetaObj * np.pi / 180.  # Converting degrees to radian

    # Creating empty mask
    imgMsk = np.zeros_like(imgData)

    # Filling the mask
    if np.int(xObj.size) == 0:
        print("ellipticalMask: no mask created")
    elif np.int(xObj.size) == 1:
        posObj = [xObj, yObj]
    ellObj = EllipticalAperture(posObj, aObj, bObj, theta=thetaObj_rad)
    ellMsk = ellObj.to_mask(method='center')[0].to_image(shape=imgData.shape)
    imgMsk = imgMsk + ellMsk
    else:
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



