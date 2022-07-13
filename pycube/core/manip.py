import numpy as np
import sep
import astropy
import copy
import matplotlib.pyplot as plt
from IPython import embed

from astropy.stats import sigma_clipped_stats
import astropy.coordinates as coord
import astropy.units as u

from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from photutils import EllipticalAperture

from astroquery.irsa_dust import IrsaDust

from extinction import apply
from pycube.instruments import vlt_muse
from pycube import cubeClass


def data_and_variance(datacontainer):
    """Function to assign variables to the data and variance of an IFUcube object.

    Parameters
    ----------
    datacontainer : IFUcube class object
        The original data file read in from cubeClass.py

    Returns
    -------
    datacube : np.array
        The 3D cube array of the data in the file
    statcube : np.array
        The 3D cube array of the variance in the file
        If no variance exists, returns as None
    """
    datacube = np.copy(datacontainer.data.data)

    try:
        statcube = np.copy(datacontainer.stat.data)
    except ValueError:
        print("No variance file detected.")
        statcube = None

    return datacube, statcube

def nicePlot():
    """Universal plotting parameters in place for debug outputs of functions
    """

    print("nicePlot: Setting rcParams")
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.direction"] = 'in'
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["xtick.labelsize"] = 30
    plt.rcParams["ytick.labelsize"] = 30
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["axes.labelsize"] = 25
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.markeredgewidth"] = 3
    plt.rcParams["patch.linewidth"] = 5
    plt.rcParams["hatch.linewidth"] = 5
    plt.rcParams["font.size"] = 30
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.handletextpad"] = 1


def find_sigma(data):
    """Simple expression to calculate Sigma quickly. Taking square root of median value of an array of values.

    Parameters
    ----------
    data : np.array
        2D array of interest to generate sigma value

    Returns
    -------
    float
        Sigma value of given 2D array, ignoring NaNs
    """
    return np.sqrt(np.nanmedian(data))


def channel_array(datacontainer, channel):
    """Given a datacontainers, and a channel (x,y,z), creates a  numpy array of values of channel range

    Parameters
    ----------
    datacontainer : np.array
        3D datacontainer ~ .Fits file
    channel : str
        Dimension name (x or X, y or Y, z or Z)

    Returns
    -------
    np.array
        Range array of length of given dimension
    """

    z_max, y_max, x_max = np.shape(datacontainer)
    if channel == 'x' or channel == 'X':
        channel_range = np.arange(0, x_max, 1, int)
    elif channel == 'y' or channel == 'Y':
        channel_range = np.arange(0, y_max, 1, int)
    elif channel == 'z' or channel == 'Z':
        channel_range = np.arange(0, z_max, 1, int)
    else:
        print("create_array: please enter a valid dimension to create from..")
        print("Valid inputs: 'x', 'X', 'y', 'Y', 'z', 'Z'.")
        return None
    return channel_range


def convert_to_wave(datacontainer, channels='z', stat=False):
    """Converts channel values in 3D MUSE data into wavelength values along z axis (wavelength).
    Specifically works with .FITS formatted MUSE data.
    Utilizes header vals ('CRVAL3' and 'CD3_3')

    Parameters
    ----------
    datacontainer : np.array
        .FITS datacontainer multi-dimensional array
    channels : str, optional
        Channel dimension (x, y, z) to create wavelength range array. default: 'z'
    stat : boolean, optional
        if True, returns a second array for variance wavelengths (default is False)
    Returns
    -------
    array
        An array of wavelength values for given channel
    """
    if stat is True:
        datacopy, statcopy = data_and_variance(datacontainer)

        data_headers = datacontainer.data.header

        data_channel_range = channel_array(datacopy, channels)
        data_wave = data_headers['CRVAL3'] + (np.array(data_channel_range) * data_headers['CD3_3'])

        stat_channel_range = channel_array(statcopy, channels)
        stat_wave = data_headers['CRVAL3'] + (np.array(stat_channel_range) * data_headers['CD3_3'])
        return np.array(data_wave, float), np.array(stat_wave, float)

    else:
        datacopy, _ = data_and_variance(datacontainer)
        data_headers = datacontainer.data.header
        data_channel_range = channel_array(datacopy, channels)
        data_wave = data_headers['CRVAL3'] + (np.array(data_channel_range) * data_headers['CD3_3'])
        return np.array(data_wave, float)


def convert_to_channel(datacontainer, channels='z'):
    """Converts wavelength values in 3D MUSE data into channel value along z axis
    Specifically works with .FITS formatted MUSE data.

    Parameters
    ----------
    datacontainer : np.array
        .FITS datacontainer multi-dimensional array
    channels : str, optional
        Channel dimension ('x', 'y', 'z' or 'X', 'Y', 'Z') to create wavelength range array. default: 'z'

    Returns
    -------
    array
        An array of channel values for given wavelength
    """
    data_headers = datacontainer.header
    channel_range = channel_array(datacontainer, channels)
    channel = np.array(channel_range) - data_headers['CRVAL3'] / data_headers['CD3_3']
    return np.array(channel, float)


def collapse_cube(datacube, min_lambda=None, max_lambda=None):
    """ Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    Parameters
    ----------
    datacube : np.array
        3D data file
    min_lambda : int, optional
        Minimum wavelength
    max_lambda : int, optional
        Maximum wavelength
    Returns
    -------
    np.array
        Condensed 2D array of 3D file.
    """

    datacopy = np.copy(datacube)
    z_max, y_max, x_max = np.shape(datacopy)
    # Checks and resets if outside boundaries of z
    if max_lambda is None or max_lambda > z_max:
        max_lambda = z_max
        print("collapse_cube : Exceeded / unspecified wavelength in data cube. "
              "Max value is set to {}".format(int(z_max)))
    if min_lambda is None or min_lambda < 0:
        min_lambda = 0
        print("collapse_cube : Invalid / unspecified minimum wavelength. Min value is set to 0")

    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)
    del datacopy
    del z_max, y_max, x_max
    return col_cube


def collapse_container(datacontainer, min_lambda=None, max_lambda=None, var_thresh=5.):
    """ Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.

    Parameters
    ----------
    datacontainer : IFUcube object
        data passed through cubeClass.py
    min_lambda : int, optional
        Minimum wavelength
    max_lambda : int, optional
        Maximum wavelength
    var_thresh : int, float, optional
        if varcontainer is None, this value determines standard deviation threshold to create variance cube from data
        (default is 5.)

    Returns
    -------
    np.array
        Condensed 2D array of 3D file.
    """

    z_max, y_max, x_max = np.shape(datacontainer.get_data())

    # Checks and sets if outside boundaries of z
    if max_lambda is None or max_lambda > z_max:
        max_lambda = z_max
        print("collapse_cube : Exceeded / unspecified wavelength in data cube. "
              "Max value is set to {}".format(int(z_max)))
    if min_lambda is None or min_lambda < 0:
        min_lambda = 0
        print("collapse_cube : Invalid / unspecified minimum wavelength. Min value is set to 0")
    col_datacube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)

    if statcopy is not None:
        statcopy = np.copy(statcopy.data)
        col_statcube = np.nansum(statcopy[min_lambda:max_lambda, :, :], axis=0)
    else:
        med_cube = np.nanmedian(col_datacube)
        std_cube = np.nanstd(col_datacube - med_cube)
        col_statcube = np.zeros_like(col_datacube) + \
                             np.nanvar(col_datacube[(col_datacube - med_cube) < (var_thresh * std_cube)])

        del med_cube
        del std_cube
    del datacopy
    del statcopy
    del z_max, y_max, x_max

    return col_datacube, col_statcube


def collapse_mean_container(datacontainer, min_lambda=None, max_lambda=None):
    """

    Inputs:
        datacontainers:
        statcube:
        min_lambda:
        max_lambda:

    Returns:

    """

    temp_collapsed_data, temp_collapsed_stat = collapse_cube(datacontainer, min_lambda, max_lambda)

    temp_collapsed_stat = 1. / temp_collapsed_stat
    badPix = np.isnan(temp_collapsed_data) | np.isnan(temp_collapsed_stat)
    # to deal with np.nan a weight of (almost) zero is given to badPix
    temp_collapsed_data[badPix] = 0.
    temp_collapsed_stat[badPix] = np.nanmin(temp_collapsed_stat) / 1000.
    collapsedDataImage, collapsedWeightsImage = np.average(temp_collapsed_data,
                                                           weights=temp_collapsed_stat,
                                                           axis=0,
                                                           returned=True)
    collapsedStatImage = 1. / collapsedWeightsImage
    plt.imshow(temp_collapsed_stat)
    plt.show()
    print("collapseCubeWMean: Images produced")

    # Deleting temporary cubes to clear up memory
    del temp_collapsed_data
    del temp_collapsed_stat
    del collapsedWeightsImage

    return collapsedDataImage, collapsedStatImage


def location(data_flat, x_position=None, y_position=None,
             semi_maj=None, semi_min=None,
             theta=0., default=10):
    """User input function to create elliptical mask of given coordinates for source in image.

    Parameters
    ----------
    data_flat : np.array
        2D collapsed image
    x_position : int, float, optional
        User given x coord of stellar object
    y_position : int, float, optional
        User given y coord of stellar object
    semi_maj : int, float, optional
        Semi-major axis (default is 10)
    semi_min : int, float, optional
        Semi-minor axis default (default is 6)
    theta : int, float, optional
        Angle for ellipse rotation around object (default is 0)
    default : int, optional
        Pixel scale (default is 10)

    Returns
    -------
    np.array
        Mask of 2D array of with user defined stellar objects
        denoted as 1 with all other elements 0
    """
    mask_array = np.zeros_like(data_flat)
    mask_shape = np.shape(data_flat)
    x_mask, y_mask = mask_shape
    object_position = (x_position, y_position)
    # if no position given..
    # results with default value. Left in place for testing
    if semi_maj is None:
        semi_maj = default
        print("location: Missing semi-major axis <- setting pixel value to {}".format(semi_maj))
    if semi_min is None:
        semi_min = default * 0.6
        print("location: Missing semi-minor axis <- setting pixel value to {}".format(semi_min))

    if x_position is None:
        print("location: no source input, no mask created")
    elif type(x_position) is int or type(x_position) is float:
        print("location: single source identified")
        theta_rad = (theta * np.pi) / 180.  # converts angle degrees to radians
        object_ellipse = EllipticalAperture(object_position, semi_maj, semi_min, theta_rad)
        ellipse_mask = object_ellipse.to_mask(method="center").to_image(shape=(x_mask, y_mask))
        mask_array += ellipse_mask
    else:
        if type(theta) is int or type(theta) is float:
            theta = np.zeros_like(x_position)
        if type(semi_min) is int or type(semi_min) is float:
            semi_min = np.full_like(x_position, semi_min)
        if type(semi_maj) is int or type(semi_maj) is float:
            semi_maj = np.full_like(x_position, semi_maj)

        #print("location: multiple sources specified, iterating through list")
        for index in range(0, len(x_position), 1):
            object_position = [x_position[index], y_position[index]]
            #print("location: Masking for object",index + 1, "at position:",object_position)
            theta_rad = (theta[index] * np.pi) / 180.  # converts angle degrees to radians
            object_ellipse = EllipticalAperture(object_position, semi_maj[index], semi_min[index], theta_rad)
            ellipse_mask = object_ellipse.to_mask(method="center").to_image(shape=(x_mask, y_mask))
            mask_array += ellipse_mask

    return np.array((mask_array > 0.), dtype=int)


def annularMask(data_flat,
                x_position,
                y_position,
                semi_maj=5.,
                semi_min=5.,
                delta_maj=2.,
                delta_min=2.,
                theta=0.):
    """Returning a mask where annuli centered in xObj and yObj with
    inner axis aObj and bObj and outer axis aObj+deltaAObj and bObj+
    deltaBObj are marked as 1.

    Parameters
    ----------
    data_flat : np.array
        data in a 2D array. The mask will have the same size
        of this.
    x_position : np.array
        x-location of the sources in pixels
    y_position : np.array
        y-location of the sources in pixels
    semi_maj : float
        inner semi-major axis in pixel
    semi_min : float
        inner semi-minor axis in pixel
    delta_maj : float
        width of the semi-major axis in pixel
    delta_min : float
        width of the semi-minor axis in pixel
    theta : float
        angle wrt the x-axis in degrees

    Returns
    -------
    np.array
        mask where sources are marked with 1 and background with 0.
        It has the same dimensions of the input imgData
    """

    # Creating mask
    imgMsk = location(data_flat, x_position, y_position,
                      semi_maj=semi_maj + delta_maj, semi_min=semi_min + delta_min,
                      theta=theta)
    imgMskInner = location(data_flat, x_position, y_position,
                           semi_maj=semi_maj, semi_min=semi_min,
                           theta=theta)
    imgMsk[(imgMskInner>0)] = np.int(0)

    # Cleaning up memory
    del imgMskInner

    return imgMsk.astype(int)


#TODO

# fix smallcube for output with datacube
def smallCube(datacontainer, min_lambda=None, max_lambda=None):
    """Given header, DATA, and STAT the macro cut the cube
    between minChannel and maxChannel. It also updates the
    wavelength information to be conformed to the new size.
    Note that minChannel and maxChannel can be given both
    as wavelength and as channels. If the input value is
    <3000. it will be assumed to be channel number, otherwise
    wavelength in Angstrom will be considered.

    Parameters
    ----------
    datacontainer : HDUList
        astropy .fits file
    min_lambda : int
        min channel to create collapsed image
    max_lambda : int
        max channel to create collapsed image

    Returns
    -------
    headFullSmall : hdu header
        primary header
    headDataSmall : hdu header
        fits header for DATA with corrected CRVAL3
    dataDataSmall : np.array
        data in a 3D array trim from minChannel to maxChannel
    headStatSmall : hdu header
        fits header for STAT with corrected CRVAL3
    dataStatSmall : np.array
        variance in a 3D array trim from minChannel to maxChannel
    """
    headFull = datacontainer.primary.header
    headData = datacontainer.data.header
    dataData = datacontainer.data.data
    headStat = datacontainer.stat.header
    dataStat = datacontainer.stat.data
    # Check for the size of the cube
    zMax, yMax, xMax = np.shape(dataData)

    # Setting min and max channels for collapsing
    if (min_lambda is not None) & (max_lambda is not None):
        minChannelSort = np.min([min_lambda, max_lambda])
        maxChannelSort = np.max([min_lambda, max_lambda])
        min_lambda, max_lambda = minChannelSort, maxChannelSort
        del minChannelSort
        del maxChannelSort
    # set values in case of Nones
    if min_lambda is None:
        print("smallCube: minChannel set to 0")
        min_lambda = np.int(0)
    if max_lambda is None:
        print("smallCube: maxChannel set to {}".format(np.int(zMax)))
        max_lambda = np.int(zMax)
    # If input values are larger than 3000., the macro converts from
    # wavelength in ang. to channel number.
    if min_lambda>3000.:
        print("smallCube: Converting min wavelength in Ang. to channel number")
        min_lambda = convert_to_channel(datacontainer.data)
    else:
        min_lambda = np.int(min_lambda)
    if max_lambda>3000.:
        print("smallCube: Converting Max wavelength in Ang. to channel number")
        max_lambda = convert_to_channel(datacontainer.data)
    else:
        max_lambda = np.int(max_lambda)
    # Check for upper and lower limits
    if min_lambda < 0:
        print("smallCube: Negative value for minChannel set to 0")
        min_lambda = np.int(0)
    if max_lambda > (zMax + 1):
        print("smallCube: maxChannel is outside the cube size. Set to {}".format(np.int(zMax)))
        max_lambda = np.int(zMax)

    smallCubeCRVAL3 = np.float(convert_to_wave(datacontainer.data.data))
    print("smallCube: Creating smaller cube")
    print("           The old pivot wavelength was {}".format(headData['CRVAL3']))
    print("           The new pivot wavelength is {}".format(smallCubeCRVAL3))
    # Header
    headFullSmall = copy.deepcopy(headFull)
    # DATA
    headDataSmall = copy.deepcopy(headData)
    headDataSmall['CRVAL3'] = smallCubeCRVAL3
    dataDataSmall = np.copy(dataData[min_lambda:max_lambda, :, :])
    # STAT
    headStatSmall = copy.deepcopy(headStat)
    headStatSmall['CRVAL3'] = smallCubeCRVAL3
    dataStatSmall = np.copy(dataStat[min_lambda:max_lambda, :, :])

    return headFullSmall, headDataSmall, dataDataSmall, headStatSmall, dataStatSmall

def dust_correction(datacontainer):
    """Function queries the IRSA dust map for E(B-V) value and
    returns a reddening array. Works along z-axis of datacontainer
    http://irsa.ipac.caltech.edu/applications/DUST/
    The query return E(B_V) from SFD (1998). This will be converted
    to the S&F (2011) one using:
    E(B-V)S&F =  0.86 * E(B-V)SFD

    Parameters
    ----------
    datacontainer : np.array
        3D .fits file array / IFU cube object

    Returns
    -------
    reddata, redstat : np.array
        dust-corrected reddened cubes for data and variance
    """
    reddata, redstat = data_and_variance(datacontainer)
    channel_range = channel_array(reddata, 'z') #creates channel along z-axis

    headers = datacontainer.primary.header
    ra = headers['RA']
    dec = headers['DEC']
    wavecube = convert_to_wave(reddata.data, 'z')  # creates wavelength value cube along z-axis

    coordinates = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
    try:
        dust_image = IrsaDust.get_images(coordinates, radius=5. * u.deg, image_type='ebv', timeout=60)[0]
    except:
        print("Increasing dust image radius to 10deg")
        dust_image = IrsaDust.get_images(coordinates, radius=10. * u.deg, image_type='ebv', timeout=60)[0]
    y_size, x_size = dust_image[0].data.shape
    ebv = 0.86 * np.mean(dust_image[0].data[y_size - 2:y_size + 2, x_size - 2:y_size + 2])
    r_v = 3.1
    av = r_v * ebv

    redcube = []
    for index in wavecube:
        reddened_wave = apply(flux=index, extinction=av)
        redcube.append(reddened_wave)
    for channel in channel_range:
        reddata[channel, :, :] *= redcube[channel]
        redstat[channel, :, :] *= redcube[channel] ** 2
    return reddata, redstat


def quickApPhotmetry(image_data,
                     image_var,
                     x_pos,
                     y_pos,
                     radius_pos=2.,
                     inner_rad=10.,
                     outer_rad=15.):
    """Performing quick circular aperture photometry on an image.
    An annular region with inner radius [inner_rad] and outer radius [outer_rad]
    will be used to estimate the background.

    Parameters
    ----------
    image_data : np.array
        image where the aperture photometry will be performed
    image_var : np.array
        variance image that will be used to calculate the errors
        on the aperture photometry
    x_pos : int, float, np.array
        x-location of the source in pixel (converts to array)
    y_pos : int, float, np.array
        y-location of the source in pixel (converts to array)
    radius_pos : np.array
        radius where to perform the aperture photometry
    inner_rad : np.array
        inner radius of the background region in pixel
    outer_rad : np.array
        outer radius of the background region in pixel

    Returns
    -------
    obj_flux, obj_err_flux, ave_flux : np.array
    """

    print("quickApPhotmetry: Performing aperture photometry")

    if np.size(x_pos) == 1:
        x_object = np.array([x_pos])
        y_object = np.array([y_pos])
    else:
        x_object = np.array(x_pos)
        y_object = np.array(y_pos)

    if np.size(radius_pos) == 1:
        radius_aperture = np.full_like(x_object, radius_pos, float)
    else:
        radius_aperture = radius_pos

    if np.size(inner_rad) == 1:
        inner_aperture = np.full_like(x_object, inner_rad, float)
    else:
        inner_aperture = inner_rad

    if np.size(outer_rad) == 1:
        outer_aperture = np.full_like(x_object, outer_rad, float)
    else:
        outer_aperture = outer_rad

    # setting up arrays
    obj_flux = np.zeros_like(x_object, float)
    obj_err_flux = np.zeros_like(x_object, float)
    ave_flux = np.zeros_like(x_object, float)

    for index in np.arange(0, np.size(x_object)):
        print("Source", index + 1, ": Aperture photometry on source at: {}, {}".format(x_object[index], y_object[index]))
        obj_pos = [x_object[index], y_object[index]]
        circle_obj = CircularAperture(obj_pos, r=radius_aperture[index])
        annulus_obj = CircularAnnulus(obj_pos, r_in=inner_aperture[index], r_out=outer_aperture[index])
        # Flux
        ap_phot = aperture_photometry(image_data, circle_obj)
        # Background
        image_datacopy = np.copy(image_data)
        bad_bg = np.min([np.nanmin(image_datacopy), -99999.])
        image_datacopy[~np.isfinite(image_datacopy)] = bad_bg
        mask_bg = annulus_obj.to_mask(method='center')
        data_bg = mask_bg.multiply(image_datacopy)
        data_bg[(np.array(mask_bg) == 0)] = bad_bg
        flat_data_bg = data_bg[data_bg > bad_bg].flatten()
        mean_bg, median_bg, sigma_bg = sigma_clipped_stats(flat_data_bg)
        # Variance
        varApPhot = aperture_photometry(image_var, circle_obj)
        # Filling arrays
        obj_flux[index] = ap_phot['aperture_sum'][0] - (median_bg * circle_obj.area)
        obj_err_flux[index] = np.power(np.array(varApPhot['aperture_sum'][0]), 0.5)
        ave_flux[index] = median_bg

    # Deleting temporary arrays to clear up memory
    del radius_aperture
    del inner_aperture
    del outer_aperture
    del obj_pos
    del x_object
    del y_object
    del image_datacopy

    return obj_flux, obj_err_flux, ave_flux

def quickSpectrum(datacontainer,
                  x_pos,
                  y_pos,
                  radius_pos=2.,
                  inner_rad=10.,
                  outer_rad=15.,
                  void_mask=None):
    """Performing quick spectrum extraction from a circular aperture on a cube.
    Background is calculated from the median of an annular aperture performing
    sigma clipping.

    Parameters
    ----------
    datacontainer : IFUcube object
        original data read into cubeClass.py
    x_pos : np.array
        x-location of the source in pixel
    y_pos: np.array
        y-location of the source in pixel
    radius_pos: np.array, optional
        radius where to perform the aperture photometry (default is 2.)
    inner_rad: np.array, optional
        inner radius of the background region in pixel (default is 10.)
    outer_rad: np.array, optional
        outer radius of the background region in pixel (default is 15.)
    void_mask : np.array, optional
        mask of possible contaminants (1->the pixel will be
        removed from the photometry) (default is None)

    Returns
    -------
    specApPhot : np.array
    specVarApPhot : np.array
    specFluxBg : np.array

    """
    datacopy, statcopy = data_and_variance(datacontainer)
    if void_mask is None:
        void_mask = np.zeros_like(datacopy[0, :, :])
    else:
        print("quickSpectrum: Using void_mask")

    specApPhot = []
    specVarApPhot = []
    specFluxBg = []
    posObj = [x_pos, y_pos]
    circObj = CircularAperture(posObj, r=radius_pos)
    annuObj = CircularAnnulus(posObj, r_in=inner_rad, r_out=outer_rad)
    zMax, yMax, xMax = datacopy.shape

    for channel in np.arange(0, zMax, 1):
        # Total flux
        tempData = np.copy(datacopy[channel, :, :])
        apPhot = aperture_photometry(tempData, circObj)
        # Background masking bad values
        tempDataBg = np.copy(datacopy[channel, :, :])
        badBg = np.min([np.nanmin(tempDataBg), -99999.])
        tempDataBg[(void_mask == 1)] = badBg
        tempDataBg[~np.isfinite(tempDataBg)] = badBg
        maskBg = annuObj.to_mask(method='center')
        dataBg = maskBg.multiply(tempDataBg)
        dataBg[(np.array(maskBg) == 0)] = badBg
        dataBg1d = dataBg[dataBg > badBg].flatten()
        meanBg, medianBg, sigmaBg = sigma_clipped_stats(dataBg1d)
        # Error
        tempStat = np.copy(statcopy[channel, :, :])
        varApPhot = aperture_photometry(tempStat, circObj)
        # Loading lists
        specApPhot.append(apPhot['aperture_sum'][0] - (medianBg * circObj.area))
        specVarApPhot.append(varApPhot['aperture_sum'][0])
        specFluxBg.append(medianBg)

    # Deleting temporary arrays to clear up memory
    del tempData
    del tempStat
    del tempDataBg

    specVarApPhot = np.array(specVarApPhot)
    specVarApPhot[~np.isfinite(specVarApPhot)] = np.nanmax(specVarApPhot)
    return np.array(specApPhot), np.power(specVarApPhot,0.5), np.array(specFluxBg)

def quickSpectrumNoBg(datacontainer,
                      x_pos,
                      y_pos,
                      radius_pos=2.):
    """Performing quick spectrum extraction from a circular aperture on a cube.

    Parameters
    ----------
    datacontainer : np.array
        data in a 3D array
    x_pos : int, float, np.array
        x-location of the source in pixel
    y_pos : int, float, np.array
        y-location of the source in pixel
    radius_pos : float
        radius where to perform the aperture photometry

    Returns
    -------
    fluxObj, errFluxObj
    """

    print("quickSpectrumNoBg: Extracting spectrum from the cube")
    datacopy, statcopy = data_and_variance(datacontainer)

    specApPhot = []
    specVarApPhot = []

    posObj  = [x_pos, y_pos]
    circObj = CircularAperture(posObj, r=radius_pos)
    zMax, yMax, xMax = np.shape(datacopy)

    for channel in np.arange(0,zMax,1):
        # Total flux
        tempData = np.copy(datacopy[channel, :, :])
        apPhot = aperture_photometry(tempData, circObj)
        # Error
        tempStat = np.copy(statcopy[channel, :, :])
        varApPhot = aperture_photometry(tempStat, circObj)
        # Loading lists
        specApPhot.append(apPhot['aperture_sum'][0])
        specVarApPhot.append(varApPhot['aperture_sum'][0])

    # Deleting temporary arrays to clear up memory
    del tempData
    del tempStat

    return np.array(specApPhot), np.power(np.array(specVarApPhot), 0.5)

def quickSpectrumNoBgMask(datacontainer,
                          statcube,
                          maskXY):
    """Performing quick spectrum extraction from an aperture given by a 2D mask
    from a cube.

    Parameters
    ----------
    datacontainer : np.array
        data in a 3D array
    statcube : np.array
        variance in a 3D array
    maskXY : np.array
        2D aperture where to perform the spectral extraction.
        Only spaxels marked as 1 are considered

    Returns
    -------
    fluxObj, errFluxObj
    """

    print("quickSpectrumNoBgMask: Extracting spectrum from the cube")
    datacopy = np.copy(datacontainer)
    statcopy = np.copy(statcube)
    zMax, yMax, xMax = datacontainer.shape
    for channel in np.arange(0, zMax, 1):
        # masking arrays
        datacopy[channel, :, :][(maskXY<1)] = 0
        statcopy[channel, :, :][(maskXY<1)] = 0

    fluxObj = np.nansum(datacopy, axis=(1,2))
    errFluxObj = np.sqrt(np.nansum(statcopy, axis=(1, 2)))

    # Deleting temporary arrays to clear up memory
    del datacopy
    del statcopy

    return fluxObj, errFluxObj

def gaussian(x, N, x0, sigma):
    """ Returns gaussian given normalization, center, and sigma.

    Parameters
    ----------
    x : np.array
        x-vector
    N, x0, sigma : float
        Normalization, center, and sigma of the gaussian

    Returns
    -------
    gauss : np.array
        The gaussian curve evaluated in x
    """

    gauss = N*np.exp(-(x-x0)**2/(2.*sigma**2))

    return np.array(gauss, float)


def statFullCube(datacontainer,
                 nSigmaExtreme=None):
    """Given a cube the macro calculate average, median, and
    sigma of all its voxels. NaNs are considered as bad pixels
    and removed.

    Parameters
    ----------
    datacontainer : np.array
        3D cube containing the voxels you want to get the
        statistic for.
    nSigmaExtreme : float, optional
        if not None, voxels with values larger than
        sigmaExtreme times the standard deviation of
        the cube will be masked (default is None)

    Returns
    -------
    cubeAverage, cubeMedian, cubeStandard : float
        average, median, and standard deviation of
        the cube.
    """

    print("statFullCube: statistic on the cube")
    dataCubeTmp = np.copy(datacontainer)

    cubeAverage = np.nanmean(dataCubeTmp)
    cubeMedian = np.nanmedian(dataCubeTmp)
    cubeStandard = np.nanstd(dataCubeTmp)

    if nSigmaExtreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extremeMask = np.zeros_like(dataCubeTmp, int)
        dataCubeTmp[~np.isfinite(dataCubeTmp)] = cubeAverage + (3.*nSigmaExtreme*cubeStandard)
        extremeMask[np.abs((dataCubeTmp-cubeAverage)/cubeStandard)>nSigmaExtreme] = 1
        cubeAverage = np.nanmean(dataCubeTmp[(extremeMask==0)])
        cubeMedian = np.nanmedian(dataCubeTmp[(extremeMask==0)])
        cubeStandard = np.nanstd(dataCubeTmp[(extremeMask==0)])
        del extremeMask

    print("statFullCube: average = {:+0.3f}".format(cubeAverage))
    print("              median  = {:+0.3f}".format(cubeMedian))
    print("              sigma   = {:+0.3f}".format(cubeStandard))

    # Cleaning up memory
    del dataCubeTmp

    return cubeAverage,  cubeMedian, cubeStandard


def statFullCubeZ(datacontainer,
                  nSigmaExtreme=None):
    """Given a cube the macro calculate average, median, and
    sigma of all its voxels along the spectral (z) axis.
    NaNs are considered as bad pixels and removed.

    Parameters
    ----------
    datacontainer : np.array
        3D cube containing the voxels you want to get the
        statistic for.
    nSigmaExtreme : np.array, optional
        if not None, voxels with values larger than
        sigmaExtreme times the standard deviation of
        the cube will be masked (default is None)

    Returns
    -------
    cubeAverageZ, cubeMedianZ, cubeStandardZ : np.arrays
        average, median, and standard deviation of
        the cube along the spectral axis.
    """

    print("statFullCubeZ: statistic on the cube")
    dataCubeTmp = np.copy(datacontainer)

    cubeAverageZ = np.nanmean(dataCubeTmp, axis=(1,2))
    cubeMedianZ = np.nanmedian(dataCubeTmp, axis=(1,2))
    cubeStandardZ = np.nanstd(dataCubeTmp, axis=(1,2))

    if nSigmaExtreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extremeMask = np.zeros_like(dataCubeTmp, int)
        zMax, yMax, xMax = np.shape(dataCubeTmp)
        for channel in np.arange(0, zMax):
            dataCubeTmp[channel, :, :][~np.isfinite(dataCubeTmp[channel, :, :])] = cubeAverageZ[channel] + (3.* nSigmaExtreme * cubeStandardZ[channel])
            extremeMask[channel, :, :][np.abs((dataCubeTmp[channel, :, :]-cubeAverageZ[channel])/cubeStandardZ[channel])> nSigmaExtreme] = 1
        dataCubeTmp[(extremeMask==1)] = np.nan
        cubeAverageZ = np.nanmean(dataCubeTmp, axis=(1,2))
        cubeMedianZ = np.nanmedian(dataCubeTmp, axis=(1,2))
        cubeStandardZ = np.nanstd(dataCubeTmp, axis=(1,2))
        del extremeMask

    # Cleaning up memory
    del dataCubeTmp

    return cubeAverageZ, cubeMedianZ, cubeStandardZ

def distFromPixel(zPix1, yPix1, xPix1,
                  zPix2, yPix2, xPix2):
    """ Given a pixel (1) and a set of locations
    (2), the macro returns the euclidean distance
    from (2) to (1)

    Parameters
    ----------
    zPix1, yPix1, xPix1 : float
        location of the pixel from which calculate the
        distances
    zPix2, yPix2, xPix2 : np.array
        location of the pixels for which the distances
        from zPix1, yPix1, xPix1 will be calculated

    Returns
    -------
    dist : np.array
        distance from (1) to (2)
    """

    zDist = np.power(zPix1-zPix2, 2.)
    yDist = np.power(yPix1-yPix2, 2.)
    xDist = np.power(xPix1-xPix2, 2.)

    dist = np.sqrt(zDist+yDist+xDist)

    return dist

def celestialToPixel(datacontainer, ra, dec):
    """Reads in a datacube and user input of an object in RA and DEC.
    Returns to user the converted X, Y position of the object
    Utilizes Header data in the fits file of the datacube entered.
    (specific)

    Parameters
    ----------
    datacontainer : np.array
        3D array
    ra : float
        Right ascension of object in degrees
    dec : float
        Declination of object in degrees

    Returns
    -------
    float
        x_pos X pixel coordinate of object
        y_pos Y pixel coordinate of object
    """
    headers = datacontainer.data.header

    raRef = headers['CRVAL1']
    raConversion = headers['CD1_1']
    xRef = headers['CRPIX1']

    decRef = headers['CRVAL2']
    decConversion = headers['CD2_2']
    yRef = headers['CRPIX2']

    raDif = ra - raRef
    decDif = dec - decRef
    x_pos = (raDif/raConversion) + xRef
    y_pos = (decDif/decConversion) + yRef
    return x_pos, y_pos


