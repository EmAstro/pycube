import numpy as np
import sep
from photutils import EllipticalAperture

def find_sigma(array):

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

# Finish writing. use np.meshgrid <-- Ryan
def location(datacube, x_position, y_position):
    """
    Inputs:
        datacube: 2D collapsed image
        x_position: User given x coord of stellar object
        y_position: User given y coord of stellar object
    Returns:
         2D array of x,y position denoted as 1 with all other elements 0
    """


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



# working on setting this to be user defined.
def findSources(datacube,statcube,
                sigDetect=3.501,
                minArea=16.):

    print("findSources: Starting sources detection")
    print("findSources: Creating background model")

    bg_median = np.nanmedian(datacube)
    bg_sigma = find_sigma(statcube)
    bg_mask = np.zeros_like(datacube)

    bg_mask[(np.abs(datacube - bg_median) > 7. * bg_sigma)] = int(1)
    img_bg = sep.Background(datacube, mask=bg_mask,
                            bw=64., bh=64., fw=5., fh=5.)

    img_data_no_bg = np.copy(datacube) - img_bg
    print("findSources: Searching sources {}-sigma above noise".format(sigDetect))
    all_objects = sep.extract(img_data_no_bg, sigDetect,
                            var=statcube,
                            minarea=minArea,
                            filter_type='matched',
                            gain=1.1,
                            clean=True,
                            deblend_cont=0.3,
                            filter_kernel=None)
    # Sorting sources by flux at the peak
    index_by_flux = np.argsort(all_objects['peak'])[::-1]
    all_objects = all_objects[index_by_flux]
    good_sources = all_objects['flag'] < 1
    xPix = np.array(all_objects['x'][good_sources])
    yPix = np.array(all_objects['y'][good_sources])
    aPix = np.array(all_objects['a'][good_sources])
    bPix = np.array(all_objects['b'][good_sources])
    angle = np.array(all_objects['theta'][good_sources]) * 180. / np.pi
    print("findSources: {} good sources detected".format(np.size(xPix)))
    # Deleting temporary images to clear up memory
    del img_bg
    del img_data_no_bg
    del good_sources
    print(xPix, yPix, aPix, bPix, angle, all_objects)

def source(datacube, ra, dec, z):
    """

    Args:
        datacube:
        ra:
        dec:
        z:

    Returns:

    """



