import extinction
import numpy as np
import copy
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
import astropy.coordinates as coord
import astropy.units as u

from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from photutils import EllipticalAperture

from astroquery.irsa_dust import IrsaDust
from pycube import msgs

CHANNELVALS=['x','X','y','Y','z','Z']

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


def channel_array(datacube, channel):
    """Given a datacontainers, and a channel (x,y,z), creates a  numpy array of values of channel range

    Parameters
    ----------
    datacube : np.array
        3D datacube to create range of channel values
    channel : str
        Dimension to create array from

    Returns
    -------
    np.array
        Range array of length of given dimension
    """
    if type(channel) != str:
        raise TypeError('Please use a string when specifying channel')
    z_max, y_max, x_max = np.shape(datacube)
    if channel == 'x' or channel == 'X':
        channel_range = np.arange(0, x_max, 1, int)
    elif channel == 'y' or channel == 'Y':
        channel_range = np.arange(0, y_max, 1, int)
    elif channel == 'z' or channel == 'Z':
        channel_range = np.arange(0, z_max, 1, int)
    else:
        raise ValueError('correct input for function is one of the following:', CHANNELVALS)
    del z_max, y_max, x_max
    return channel_range


def convert_to_wave(datacontainer, datacube, channels='z'):
    """Converts channel values in 3D MUSE data into wavelength values along z axis (wavelength).
    Specifically works with .FITS formatted MUSE data.
    Utilizes header vals ('CRVAL3' and 'CD3_3')

    Parameters
    ----------
    datacontainer : IFUcube object
        data initialized in cubeClass.py
    datacube : np.array
        3D array to be converted to wavelengths
    channels : str, optional
        Channel dimension ('x', 'y', 'z' or 'X', 'Y', 'Z') to create wavelength range array (default: 'z')
    Returns
    -------
    np.array
        An array of wavelength values for given channel
    """

    data_headers = datacontainer.get_data_header()
    data_channel_range = channel_array(datacube, channels)
    data_wave = data_headers['CRVAL3'] + (np.array(data_channel_range) * data_headers['CD3_3'])
    return np.array(data_wave, float)


def convert_to_channel(datacontainer, datacube, channels='z'):
    """Converts wavelength values in 3D MUSE data into channel value along z axis
    Specifically works with .FITS formatted MUSE data.

    Parameters
    ----------
    datacontainer : IFUcube
        data initialized in cubeClass.py
    datacube : np.array
        3D data array to be converted to channels
    channels : str, optional
        Channel dimension ('x', 'y', 'z' or 'X', 'Y', 'Z') to create channel range array (default: 'z')

    Returns
    -------
    np.array
        An array of channel values for given wavelength
    """

    data_headers = datacontainer.get_data_header()
    channel_range = channel_array(datacube, channels)
    channel = np.array(channel_range) - data_headers['CRVAL3'] / data_headers['CD3_3']
    return np.array(channel, float)


def collapse_cube(datacube, min_lambda=None, max_lambda=None, mask_z=None):
    """ Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    If mask_z is specified, the function will remove all channels masked as 1.
    Parameters
    ----------
    datacube : np.array
        3D data file
    min_lambda : int, optional
        Minimum wavelength
    max_lambda : int, optional
        Maximum wavelength
    mask_z : np.array, optional
        range of z-axis values that the user can mask in the datacube
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
    if mask_z is not None:
        datacopy[mask_z, :, :] = np.nan
    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)
    del datacopy
    del z_max, y_max, x_max
    return col_cube


def collapse_mean_cube(datacube, statcube, min_lambda=None, max_lambda=None, mask_z=None):
    """Given 3D arrays of data and variance, this function collapses it along the
    z-axis between min_lambda and max_lambda. If maskZ is given, channels masked
    as 1 (or True) are removed. The macro uses the stat information to perform a
    weighted mean along the velocity axis. In other words, each spaxel of the resulting
    image will be the weighted mean spectrum of that spaxels along the wavelengths.

    Parameters
    ----------
    datacube : np.array
        3D data file to be collapsed
    statcube : np.array
        3D variance file to be collapsed
    min_lambda : int
        minimum wavelength to collapse cube
    max_lambda : int
        maximum wavelength to collapse cube
    mask_z : np.array
        if specified, channels masked as 1 will be removed

    Returns
    -------
    np.arrays
        collapsed data and collapsed variance cubes

    """

    temp_collapsed_data = collapse_cube(datacube, min_lambda,
                                        max_lambda, mask_z=mask_z)
    temp_collapsed_stat = collapse_cube(statcube, min_lambda,
                                        max_lambda, mask_z=mask_z)

    temp_collapsed_stat = 1. / temp_collapsed_stat
    bad_pix = np.isnan(temp_collapsed_data) | np.isnan(temp_collapsed_stat)
    # to deal with np.nan a weight of (almost) zero is given to bad_pix
    temp_collapsed_data[bad_pix] = 0.
    temp_collapsed_stat[bad_pix] = np.nanmin(temp_collapsed_stat) / 1000.
    collapsed_data, collapsed_weights = np.average(temp_collapsed_data,
                                                   weights=temp_collapsed_stat,
                                                   axis=0,
                                                   returned=True)
    collapsed_stat = 1. / collapsed_weights
    plt.imshow(temp_collapsed_stat)
    plt.show()
    print("collapse_mean_cube: Images produced")

    # preserve memory
    del temp_collapsed_data
    del temp_collapsed_stat
    del collapsed_weights

    return collapsed_data, collapsed_stat


def collapse_container(datacontainer, min_lambda=None, max_lambda=None, mask_z=None, var_thresh=5.):
    """ Given an IFUcube, this function collapses along the z-axis given a range of values.
    If mask_z is specified, it will remove values masked as 1. If no variance (stat) data exists,
    it will be generated from the data.

    Parameters
    ----------
    datacontainer : IFUcube object
        data passed through cubeClass.py
    min_lambda : int, optional
        Minimum wavelength
    max_lambda : int, optional
        Maximum wavelength
    mask_z : np.array, optional
        range of z-axis values that the user can mask in the datacube
    var_thresh : int, float, optional
        if no variance in container, this value determines standard deviation threshold
         to create variance cube from data (default is 5.)

    Returns
    -------
    np.array
        Condensed 2D array of 3D file.
    """

    datacopy, statcopy = datacontainer.get_data_stat()
    z_max, y_max, x_max = np.shape(datacopy)

    # Checks and sets if outside boundaries of z
    if max_lambda is None or max_lambda > z_max:
        max_lambda = z_max
        print("collapse_container: Exceeded / unspecified wavelength in data cube. "
              "Max value is set to {}".format(int(z_max)))
    if min_lambda is None or min_lambda < 0:
        min_lambda = 0
        print("collapse_container: Invalid / unspecified minimum wavelength. Min value is set to 0")
    if mask_z is not None:
        datacopy[mask_z, :, :] = np.nan
    col_datacube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)

    if statcopy is not None:
        if mask_z is not None:
            statcopy[mask_z, :, :] = np.nan
        fin_col_statcube = np.nansum(statcopy[min_lambda:max_lambda, :, :], axis=0)
    else:
        med_cube = np.nanmedian(col_datacube)
        std_cube = np.nanstd(col_datacube - med_cube)
        col_statcube = np.zeros_like(col_datacube) + \
                       np.nanvar(col_datacube[(col_datacube - med_cube) < (var_thresh * std_cube)])
        if mask_z is not None:
            col_statcube[mask_z, :, :] = np.nan
        fin_col_statcube = np.nansum(col_statcube[min_lambda:max_lambda, :, :], axis=0)

        del med_cube
        del std_cube
    del datacopy
    del statcopy
    del z_max, y_max, x_max

    return col_datacube, fin_col_statcube


def collapse_mean_container(datacontainer, min_lambda=None, max_lambda=None, mask_z=None,threshold=5.):
    """Given an IFUcube, collapse it along the z-axis between min_lambda and
    max_lambda. If maskZ is given, channels masked as 1 (or True) are removed. The
    macro uses the stat information to perform a weighted mean along the velocity
    axis. In other words, each spaxel of the resulting image will be the weighted
    mean spectrum of that spaxels along the wavelengths.

    Parameters
    ----------
    datacontainer : IFUcube object
        data passed through cubeClass.py
    min_lambda : int, optional
        minimum wavelength to collapse cube
    max_lambda : int, optional
        maximum wavelength to collapse cube
    mask_z : np.array, optional
        if specified, will remove channels masked with 1
    threshold : int, float, optional
        if no variance file in data, variance is created from the data. Parameter changes threshold of standard
        deviation for the creation of this variance (default is 5.)

    Returns
    -------
    np.array
        collapsed data and variance

    """

    temp_collapsed_data, temp_collapsed_stat = collapse_container(datacontainer, min_lambda, max_lambda,
                                                                  mask_z=mask_z, var_thresh=threshold)

    temp_collapsed_stat = 1. / temp_collapsed_stat
    bad_pix = np.isnan(temp_collapsed_data) | np.isnan(temp_collapsed_stat)
    # to deal with np.nan a weight of (almost) zero is given to badPix
    temp_collapsed_data[bad_pix] = 0.
    temp_collapsed_stat[bad_pix] = np.nanmin(temp_collapsed_stat) / 1000.
    collapsed_data, collapsed_weights = np.average(temp_collapsed_data,
                                                   weights=temp_collapsed_stat,
                                                   axis=0,
                                                   returned=True)
    collapsed_stat = 1. / collapsed_weights
    plt.imshow(temp_collapsed_stat)
    plt.show()
    print("collapse_mean_container: Images produced")

    # Deleting temporary cubes to clear up memory
    del temp_collapsed_data
    del temp_collapsed_stat
    del collapsed_weights

    return collapsed_data, collapsed_stat


def location(data_flat, x_position=None, y_position=None,
             semi_maj=None, semi_min=None,
             theta=0.):
    """User input function to create elliptical mask of given coordinates for a source in an image.
    Works also for multiple position inputs to create a mask of all sources present.

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
    if semi_maj is None:
        semi_maj = 10
        print("location: Missing semi-major axis <- setting pixel value to {}".format(semi_maj))
    if semi_min is None:
        semi_min = 6
        print("location: Missing semi-minor axis <- setting pixel value to {}".format(semi_min))

    if x_position is None:
        print("location: Please specify location of the object")
    elif type(x_position) is int or type(x_position) is float:
        print("location: single source identified")
        theta_rad = (theta * np.pi) / 180.  # converts angle degrees to radians
        object_ellipse = EllipticalAperture(object_position, semi_maj, semi_min, theta_rad)
        ellipse_mask = object_ellipse.to_mask(method="center").to_image(shape=(x_mask, y_mask))
        mask_array += ellipse_mask
    else:
        # accounts for single input of angle and semi-diameter values
        # creates full arrays of the specified value to apply to all objects
        if type(theta) is int or type(theta) is float:
            theta = np.zeros_like(x_position)
        if type(semi_min) is int or type(semi_min) is float:
            semi_min = np.full_like(x_position, semi_min)
        if type(semi_maj) is int or type(semi_maj) is float:
            semi_maj = np.full_like(x_position, semi_maj)

        # masking for multiple source input (array of (x,y))
        for index in range(0, len(x_position), 1):
            object_position = [x_position[index], y_position[index]]
            theta_rad = (theta[index] * np.pi) / 180.  # converts angle degrees to radians
            object_ellipse = EllipticalAperture(object_position, semi_maj[index], semi_min[index], theta_rad)
            ellipse_mask = object_ellipse.to_mask(method="center").to_image(shape=(x_mask, y_mask))
            mask_array += ellipse_mask

    # preserve memory
    del object_ellipse
    del ellipse_mask
    del object_position
    del x_mask
    del y_mask
    return np.array((mask_array > 0.), int)


def annularMask(data_flat,
                x_pos,
                y_pos,
                semi_maj=5.,
                semi_min=5.,
                delta_maj=2.,
                delta_min=2.,
                theta=0.):
    """Returning a mask where annuli centered in x_pos and y_pos with
    inner axis (semi_maj, semi_min) and with
    outer axis (semi_maj + delta_maj, semi_min + delta_min) are marked as 1.

    Parameters
    ----------
    data_flat : np.array
        data in a 2D array. The mask will have the same size
        of this.
    x_pos : np.array
        x-location of the sources in pixels
    y_pos : np.array
        y-location of the sources in pixels
    semi_maj : float, optional
        inner semi-major axis in pixel (default is 5.)
    semi_min : float, optional
        inner semi-minor axis in pixel (default is 5.)
    delta_maj : float, optional
        width of the semi-major axis in pixel (default is 2.)
    delta_min : float, optional
        width of the semi-minor axis in pixel (default is 2.)
    theta : float, optional
        angle wrt the x-axis in degrees (default is 0.)

    Returns
    -------
    np.array
        mask where sources are marked with 1 and background with 0.
        It has the same dimensions of the input datacube
    """

    # Creating mask
    img_msk = location(data_flat, x_pos, y_pos,
                       semi_maj=semi_maj + delta_maj, semi_min=semi_min + delta_min,
                       theta=theta)
    inner_img_mask = location(data_flat, x_pos, y_pos,
                              semi_maj=semi_maj, semi_min=semi_min,
                              theta=theta)
    img_msk[(inner_img_mask > 0)] = 0

    # Cleaning up memory
    del inner_img_mask

    return img_msk.astype(int)


def smallCube(datacontainer, min_lambda=None, max_lambda=None):
    """Given an IFUcube, this function collapses along min_lambda
    and max_lambda. It also updates the wavelength information to
    be conformed to the new size. Note that min_lambda and max_lambda
    can be given both as wavelength and as channels. If the input value is
    <3000. it will be assumed to be channel number, otherwise
    wavelength in Angstrom will be considered.

    Parameters
    ----------
    datacontainer : IFUcube object
        data intialized in cubeClass.py
    min_lambda : int, optional
        min channel to create collapsed image
    max_lambda : int, optional
        max channel to create collapsed image

    Returns
    -------
    s_primary_headers : hdu header
        primary header
    s_data_headers : hdu header
        fits header for DATA with corrected CRVAL3
    s_datacopy : np.array
        data in a 3D array trim from min_lambda to max_lambda
    s_stat_headers : hdu header
        fits header for STAT with corrected CRVAL3
    s_statcopy : np.array
        variance in a 3D array trim from min_lambda to max_lambda
    """

    datacopy, statcopy = datacontainer.get_data_stat()
    primary_headers = datacontainer.get_primary()
    data_headers, stat_headers = datacontainer.get_headers()
    # Check for the size of the cube
    z_max, y_max, x_max = np.shape(datacopy)

    # Setting min and max channels for collapsing
    if (min_lambda is not None) & (max_lambda is not None):
        min_channel_sort = np.min([min_lambda, max_lambda])
        max_channel_sort = np.max([min_lambda, max_lambda])
        min_lambda, max_lambda = min_channel_sort, max_channel_sort
        del min_channel_sort
        del max_channel_sort
    # set values in case of Nones
    if min_lambda is None:
        print("smallCube: min_lambda set to 0")
        min_lambda = 0
    if max_lambda is None:
        print("smallCube: max_lambda set to {}".format(int(z_max)))
        max_lambda = int(z_max)
    # If input values are larger than 3000., the macro converts from
    # wavelength in ang. to channel number.
    if min_lambda > 3000.:
        print("smallCube: Converting min wavelength in Ang. to channel number")
        min_lambda = int((np.array(min_lambda - data_headers['CRVAL3']) / data_headers['CD3_3']))
    else:
        min_lambda = int(min_lambda)
    if max_lambda > 3000.:
        print("smallCube: Converting Max wavelength in Ang. to channel number")
        max_lambda = int((np.array(max_lambda - data_headers['CRVAL3']) / data_headers['CD3_3']))
    else:
        max_lambda = int(max_lambda)
    # Check for upper and lower limits
    if min_lambda < 0:
        print("smallCube: Negative value for min_lambda set to 0")
        min_lambda = int(0)
    if max_lambda > (z_max + 1):
        print("smallCube: max_lambda is outside the cube size. Set to {}".format(int(z_max)))
        max_lambda = int(z_max)

    small_cube_CRVAL3 = float(np.array(data_headers['CRVAL3'] + (np.array(min_lambda) * data_headers['CD3_3'])))
    print("smallCube: Creating smaller cube")
    print("           The old pivot wavelength was {}".format(data_headers['CRVAL3']))
    print("           The new pivot wavelength is {}".format(small_cube_CRVAL3))
    # Header
    s_primary_headers = copy.deepcopy(primary_headers)
    # DATA
    s_data_headers = copy.deepcopy(data_headers)
    s_data_headers['CRVAL3'] = small_cube_CRVAL3
    s_datacopy = np.copy(datacopy[min_lambda:max_lambda, :, :])
    # STAT
    s_stat_headers = copy.deepcopy(stat_headers)
    s_stat_headers['CRVAL3'] = small_cube_CRVAL3
    s_statcopy = np.copy(statcopy[min_lambda:max_lambda, :, :])

    return s_primary_headers, s_data_headers, s_datacopy, s_stat_headers, s_statcopy


def dust_correction(datacontainer):
    """Function queries the IRSA dust map for E(B-V) value and
    returns a reddening array. Works along z-axis of datacube
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
    reddata, redstat = datacontainer.get_data_stat()
    channel_range = channel_array(reddata, 'z')  # creates channel along z-axis

    headers = datacontainer.get_primary()
    ra = headers['RA']
    dec = headers['DEC']
    wavecube = convert_to_wave(datacontainer, reddata)  # creates wavelength value cube along z-axis

    coordinates = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
    try:
        dust_image = IrsaDust.get_images(coordinates, radius=5. * u.deg, image_type='ebv', timeout=60)[0]
    except:
        print("Increasing dust image radius to 10deg")
        dust_image = IrsaDust.get_images(coordinates, radius=10. * u.deg, image_type='ebv', timeout=60)[0]
    y_size, x_size = dust_image[0].data.shape
    ebv = 0.86 * np.mean(dust_image[0].data[y_size - 2:y_size + 2, x_size - 2:x_size + 2])
    r_v = 3.1
    av = r_v * ebv

    reddened_wave = extinction.fm07(wavecube, av)
    for channel in channel_range:
        reddata[channel, :, :] *= reddened_wave[channel]
        redstat[channel, :, :] *= (reddened_wave[channel] ** 2)
    """
    redcube = []
    for index in wavecube:
        reddened_wave = extinction.fm07(index, av)
        redcube.append(reddened_wave)
    for channel in channel_range:
        reddata[channel, :, :] *= redcube[channel]
        redstat[channel, :, :] *= (redcube[channel] ** 2)
    
    del redcube
    """
    return reddata, redstat


def quickApPhotmetry(datacopy,
                     statcopy,
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
    datacopy : np.array
        image where the aperture photometry will be performed
    statcopy : np.array
        variance image that will be used to calculate the errors
        on the aperture photometry
    x_pos : int, float, np.array
        x-location of the source in pixel (converts to array)
    y_pos : int, float, np.array
        y-location of the source in pixel (converts to array)
    radius_pos : np.array, optional
        radius where to perform the aperture photometry (default is 2.)
    inner_rad : np.array, optional
        inner radius of the background region in pixel (default is 10.)
    outer_rad : np.array, optional
        outer radius of the background region in pixel (default is 15.)

    Returns
    -------
    obj_flux, obj_err_flux, ave_flux : np.array
        flux of object, sigma of flux, and average flux values
    """

    print("quickApPhotmetry: Performing aperture photometry")

    # converts any input into a np.array
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
        print("Source", index + 1,
              ": Aperture photometry on source at: {}, {}".format(x_object[index], y_object[index]))
        obj_pos = [x_object[index], y_object[index]]
        circle_obj = CircularAperture(obj_pos, r=radius_aperture[index])
        annulus_obj = CircularAnnulus(obj_pos, r_in=inner_aperture[index], r_out=outer_aperture[index])
        # Flux
        ap_phot = aperture_photometry(datacopy, circle_obj)
        # Background
        image_datacopy = np.copy(datacopy)
        bad_bg = np.min([np.nanmin(image_datacopy), -99999.])
        image_datacopy[~np.isfinite(image_datacopy)] = bad_bg
        mask_bg = annulus_obj.to_mask(method='center')
        data_bg = mask_bg.multiply(image_datacopy)
        data_bg[(np.array(mask_bg) == 0)] = bad_bg
        flat_data_bg = data_bg[data_bg > bad_bg].flatten()
        mean_bg, median_bg, sigma_bg = sigma_clipped_stats(flat_data_bg)
        # Variance
        var_ap_phot = aperture_photometry(statcopy, circle_obj)
        # Filling arrays
        obj_flux[index] = ap_phot['aperture_sum'][0] - (median_bg * circle_obj.area)
        obj_err_flux[index] = np.power(np.array(var_ap_phot['aperture_sum'][0]), 0.5)
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


def quickApPhotmetryNoBg(datacopy,
                         statcopy,
                         x_pos,
                         y_pos,
                         obj_rad=2.):
    """Performing quick circular aperture photometry on an image
    without background subtraction

    Parameters
    ----------
    datacopy : np.array
        image where the aperture photometry will be performed
    statcopy : np.array
        variance image that will be used to calculate the errors
        on the aperture photometry
    x_pos : np.array
        x-location of the source in pixel
    y_pos : np.array
        y-location of the source in pixel
    obj_rad : np.array, optional
        radius where to perform the aperture photometry (default is 2.)
    Returns
    -------
    fluxObj, errFluxObj : np.arrays
        fluxes and errors of the sources derived from datacopy
        and statcube
    """
    print("quickApPhotmetryNoBg: Performing aperture photometry")

    if np.size(x_pos) == 1:
        x_object = np.array([x_pos])
        y_object = np.array([y_pos])
    else:
        x_object = np.array(x_pos)
        y_object = np.array(y_pos)

    if np.size(obj_rad) == 1:
        r_aperture = np.full_like(x_object, obj_rad, float)
    else:
        r_aperture = np.array(obj_rad)

    # setting up empty arrays
    flux_obj = np.zeros_like(x_object, float)
    err_flux_obj = np.zeros_like(x_object, float)

    # filling arrrays
    print("quickApPhotmetryNoBg: Aperture photometry on source:")
    for index in range(0, np.size(x_object)):
        print("                      {:03d} -> ({:03.2f},{:03.2f})".format(index, float(x_object[index]),
                                                                           float(y_object[index])))
        object_pos = [x_object[index], y_object[index]]
        circ_obj = CircularAperture(object_pos, r=r_aperture[index])
        ap_phot = aperture_photometry(datacopy, circ_obj)
        var_ap_phot = aperture_photometry(statcopy, circ_obj)
        flux_obj[index] = ap_phot['aperture_sum'][0]
        err_flux_obj[index] = np.power(np.array(var_ap_phot['aperture_sum'][0]), 0.5)

    # Deleting temporary arrays to clear up memory
    del r_aperture
    del object_pos
    del circ_obj
    del ap_phot
    del var_ap_phot

    return flux_obj, err_flux_obj


def quickSpectrum(datacontainer,
                  x_pos,
                  y_pos,
                  statcube=None,
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
    statcube : np.array, optional
        variance in 3D array if specifying a 3D array for datacube.
        Otherwise, pulls this array from IFUcube object (default is None)
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
    spec_ap_phot : np.array
        spectra of selected object
    spec_var_ap_phot : np.array
        sigma of spectra
    spec_flux_bg : np.array
        background spectral flux
    """
    # lets user pass an IFUcube object or two 3D arrays
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = datacontainer
        statcopy = np.copy(statcube)

    if void_mask is None:
        void_mask = np.zeros_like(datacopy[0, :, :])
    else:
        print("quickSpectrum: Using void_mask")

    spec_ap_phot = []
    spec_var_ap_phot = []
    spec_flux_bg = []
    object_pos = [x_pos, y_pos]

    circ_obj = CircularAperture(object_pos, r=radius_pos)
    annu_obj = CircularAnnulus(object_pos, r_in=inner_rad, r_out=outer_rad)
    z_max, y_max, x_max = datacopy.shape

    for channel in np.arange(0, z_max, 1):
        # Total flux
        temp_data = np.copy(datacopy[channel, :, :])
        ap_phot = aperture_photometry(temp_data, circ_obj)
        # Background masking bad values
        temp_data_bg = np.copy(datacopy[channel, :, :])
        bad_bg = np.min([np.nanmin(temp_data_bg), -99999.])
        temp_data_bg[(void_mask == 1)] = bad_bg
        temp_data_bg[~np.isfinite(temp_data_bg)] = bad_bg
        mask_bg = annu_obj.to_mask(method='center')
        data_bg = mask_bg.multiply(temp_data_bg)
        data_bg[(np.array(mask_bg) == 0)] = bad_bg
        data_bg1d = data_bg[data_bg > bad_bg].flatten()
        mean_bg, median_bg, sigma_bg = sigma_clipped_stats(data_bg1d)
        # Error
        temp_stat = np.copy(statcopy[channel, :, :])
        var_ap_phot = aperture_photometry(temp_stat, circ_obj)
        # Loading lists
        spec_ap_phot.append(ap_phot['aperture_sum'][0] - (median_bg * circ_obj.area))
        spec_var_ap_phot.append(var_ap_phot['aperture_sum'][0])
        spec_flux_bg.append(median_bg)

    # Deleting temporary arrays to clear up memory
    del temp_data
    del temp_stat
    del temp_data_bg

    spec_var_ap_phot = np.array(spec_var_ap_phot)
    spec_var_ap_phot[~np.isfinite(spec_var_ap_phot)] = np.nanmax(spec_var_ap_phot)
    return np.array(spec_ap_phot), np.power(spec_var_ap_phot, 0.5), np.array(spec_flux_bg)


def quickSpectrumNoBg(datacontainer,
                      x_pos,
                      y_pos,
                      statcube=None,
                      radius_pos=2.):
    """Performing quick spectrum extraction from a circular aperture on a cube.

    Parameters
    ----------
    datacontainer : np.array, IFUcube Object
        data in a 3D array or object initiated in cubeClass.py
    x_pos : int, float, np.array
        x-location of the source in pixel
    y_pos : int, float, np.array
        y-location of the source in pixel
    statcube : np.array, optional
        variance in 3D array if specifying a 3D array for datacube.
        Otherwise, pulls this array from IFUcube object (default is None)
    radius_pos : float
        radius where to perform the aperture photometry

    Returns
    -------
    spec_ap_phot : np.array
        spectra of selected object
    spec_var_ap_phot : np.array
        sigma of spectra
    """

    print("quickSpectrumNoBg: Extracting spectrum from the cube")
    # lets user pass an IFUcube object or two 3D arrays
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = np.copy(datacontainer)
        statcopy = np.copy(statcube)
    spec_ap_phot = []
    spec_var_ap_phot = []

    object_pos = [x_pos, y_pos]
    circ_obj = CircularAperture(object_pos, r=radius_pos)
    z_max, y_max, x_max = np.shape(datacopy)

    for channel in np.arange(0, z_max, 1):
        # Total flux
        temp_data = np.copy(datacopy[channel, :, :])
        ap_phot = aperture_photometry(temp_data, circ_obj)
        # Error
        temp_stat = np.copy(statcopy[channel, :, :])
        var_ap_phot = aperture_photometry(temp_stat, circ_obj)
        # Loading lists
        spec_ap_phot.append(ap_phot['aperture_sum'][0])
        spec_var_ap_phot.append(var_ap_phot['aperture_sum'][0])

    # Deleting temporary arrays to clear up memory
    del temp_data
    del temp_stat

    return np.array(spec_ap_phot), np.power(np.array(spec_var_ap_phot), 0.5)


def quickSpectrumNoBgMask(datacontainer,
                          mask_xy, statcube=None):
    """Performing quick spectrum extraction from an aperture given by a 2D mask
    from a cube.

    Parameters
    ----------
    datacontainer : IFUcube object, np.array
        data intialized in cubeClass.py or 3D array of data
    mask_xy : np.array
        2D aperture where to perform the spectral extraction.
        Only spaxels marked as 1 are considered
    statcube : np.array
        3D variance array

    Returns
    -------
    flux_obj :
        sum of object flux ignoring Nans
    err_flux_obj :
        sigma of variance
    """
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = np.copy(datacontainer)
        statcopy = np.copy(statcube)

    print("quickSpectrumNoBgMask: Extracting spectrum from the cube")
    z_max, y_max, x_max = np.shape(datacopy)
    for channel in np.arange(0, z_max, 1):
        # masking arrays
        datacopy[channel, :, :][(mask_xy < 1)] = 0
        statcopy[channel, :, :][(mask_xy < 1)] = 0

    flux_obj = np.nansum(datacopy, axis=(1, 2))
    err_flux_obj = np.sqrt(np.nansum(statcopy, axis=(1, 2)))

    # Deleting temporary arrays to clear up memory
    del datacopy
    del statcopy

    return flux_obj, err_flux_obj


def gaussian(x, norm, x0, sigma):
    """ Returns gaussian given normalization, center, and sigma.

    Parameters
    ----------
    x : np.array
        x-vector
    norm, x0, sigma : float
        Normalization, center, and sigma of the gaussian

    Returns
    -------
    gauss : np.array
        The gaussian curve evaluated in x
    """

    gauss = norm * np.exp(-(x - x0) ** 2 / (2. * sigma ** 2))

    return np.array(gauss, float)


def statFullCube(datacube,
                 n_sigma_extreme=None):
    """Given a cube the macro calculate average, median, and
    sigma of all its voxels. NaNs are considered as bad pixels
    and removed.

    Parameters
    ----------
    datacube : np.array
        3D cube containing the voxels you want to get the
        statistic for.
    n_sigma_extreme : float, optional
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
    datacopy = np.copy(datacube)

    cube_average = np.nanmean(datacopy)
    cube_median = np.nanmedian(datacopy)
    cube_standard = np.nanstd(datacopy)

    if n_sigma_extreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extreme_mask = np.zeros_like(datacopy, int)
        datacopy[~np.isfinite(datacopy)] = cube_average + (3. * n_sigma_extreme * cube_standard)
        extreme_mask[np.abs((datacopy - cube_average) / cube_standard) > n_sigma_extreme] = 1
        cube_average = np.nanmean(datacopy[(extreme_mask == 0)])
        cube_median = np.nanmedian(datacopy[(extreme_mask == 0)])
        cube_standard = np.nanstd(datacopy[(extreme_mask == 0)])
        del extreme_mask

    print("statFullCube: average = {:+0.3f}".format(cube_average))
    print("              median  = {:+0.3f}".format(cube_median))
    print("              sigma   = {:+0.3f}".format(cube_standard))

    # Cleaning up memory
    del datacopy

    return cube_average, cube_median, cube_standard


def statFullCubeZ(datacube,
                  n_sigma_extreme=None):
    """Given a cube the macro calculate average, median, and
    sigma of all its voxels along the spectral (z) axis.
    NaNs are considered as bad pixels and removed.

    Parameters
    ----------
    datacube : np.array
        3D cube containing the voxels you want to get the
        statistic for.
    n_sigma_extreme : np.array, optional
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
    datacopy = np.copy(datacube)

    cube_average_z = np.nanmean(datacopy, axis=(1, 2))
    cube_median_z = np.nanmedian(datacopy, axis=(1, 2))
    cube_standard_z = np.nanstd(datacopy, axis=(1, 2))

    if n_sigma_extreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extreme_mask = np.zeros_like(datacopy, int)
        z_max, y_max, x_max = np.shape(datacopy)
        for channel in np.arange(0, z_max):
            datacopy[channel, :, :][~np.isfinite(datacopy[channel, :, :])] = cube_average_z[channel] + (
                    3. * n_sigma_extreme * cube_standard_z[channel])
            extreme_mask[channel, :, :][np.abs(
                (datacopy[channel, :, :] - cube_average_z[channel]) / cube_standard_z[channel]) > n_sigma_extreme] = 1
        datacopy[(extreme_mask == 1)] = np.nan
        cube_average_z = np.nanmean(datacopy, axis=(1, 2))
        cube_median_z = np.nanmedian(datacopy, axis=(1, 2))
        cube_standard_z = np.nanstd(datacopy, axis=(1, 2))
        del extreme_mask

    # Cleaning up memory
    del datacopy

    return cube_average_z, cube_median_z, cube_standard_z


def distFromPixel(z_pos1, y_pos1, x_pos1,
                  z_pos2, y_pos2, x_pos2):
    """ Given a pixel (1) and a set of locations
    (2), the macro returns the euclidean distance
    from (2) to (1)

    Parameters
    ----------
    z_pos1, y_pos1, x_pos1 : float
        location of the pixel from which calculate the
        distances
    z_pos2, y_pos2, x_pos2 : np.array
        location of the pixels for which the distances
        from zPix1, y_pos1, xPix1 will be calculated

    Returns
    -------
    dist : np.array
        distance from (1) to (2)
    """

    z_dist = np.power(z_pos1 - z_pos2, 2.)
    y_dist = np.power(y_pos1 - y_pos2, 2.)
    x_dist = np.power(x_pos1 - x_pos2, 2.)

    dist = np.sqrt(z_dist + y_dist + x_dist)

    return dist


def object_coord(datacontainer, ra, dec):
    """Reads in IFUcube and user input of an object in RA and DEC.
    Returns to user the converted X, Y pixel position of the object
    Utilizes Header data in the fits file of the IFUcube entered.
    (specific)

    Parameters
    ----------
    datacontainer : IFUcube Object
        data file initialized in cubeClass.py
    ra : float
        Right ascension of object in degrees
    dec : float
        Declination of object in degrees

    Returns
    -------
    float
        x_pos X pixel coordinate of object in datacube
        y_pos Y pixel coordinate of object in datacube
    """
    headers = datacontainer.get_data_header()

    raRef = headers['CRVAL1']
    raConversion = headers['CD1_1']
    xRef = headers['CRPIX1']

    decRef = headers['CRVAL2']
    decConversion = headers['CD2_2']
    yRef = headers['CRPIX2']

    raDif = ra - raRef
    decDif = dec - decRef
    x_pos = (raDif / raConversion) + xRef
    y_pos = (decDif / decConversion) + yRef
    return x_pos, y_pos
