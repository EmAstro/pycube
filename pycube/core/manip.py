""" Module of helpful functions to aid PSF subtraction and emission searching modules in pycube
"""
import extinction
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Ellipse


from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.coordinates as coord
import astropy.units as u

from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from photutils import EllipticalAperture
from photutils import centroids
from astroquery.irsa_dust import IrsaDust

from pycube import cubeClass

from scipy import ndimage


def find_sigma(data):
    """Simple expression to calculate Sigma quickly. Taking square root of the median value of an array.

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


def channel_array(datacube,
                  channel):
    """Given a 3D data cube, and a channel (x,y,z), creates a numpy array of values of channel range

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
    channel_vals = ['x', 'X', 'y', 'Y', 'z', 'Z']
    channel = str(channel)
    z_max, y_max, x_max = np.shape(datacube)

    if channel == 'x' or channel == 'X':
        channel_range = np.arange(0, x_max, 1, int)
    elif channel == 'y' or channel == 'Y':
        channel_range = np.arange(0, y_max, 1, int)
    elif channel == 'z' or channel == 'Z':
        channel_range = np.arange(0, z_max, 1, int)
    else:
        raise ValueError('correct input for function is one of the following:', channel_vals)

    del channel_vals
    del z_max, y_max, x_max
    return channel_range


def convert_to_wave(datacontainer,
                    datacube,
                    channels='z'):
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
        Channel dimension ('x', 'y', 'z' or 'X', 'Y', 'Z') to create wavelength range array (Default is 'z')

    Returns
    -------
    np.array
        An array of wavelength values for given channel
    """
    print('DO NOT USE THIS')
    data_headers = datacontainer.get_data_header()
    data_channel_range = channel_array(datacube, channels)
    data_wave = data_headers['CRVAL3'] + (np.array(data_channel_range) * data_headers['CD3_3'])
    return np.array(data_wave, float)


def convert_to_channel(datacontainer,
                       datacube,
                       channels='z'):
    """Converts wavelength values in 3D MUSE data into channel value along z axis
    Specifically works with .FITS formatted MUSE data.

    Parameters
    ----------
    datacontainer : IFUcube
        data initialized in cubeClass.py
    datacube : np.array
        3D data array to be converted to channels
    channels : str, optional
        Channel dimension ('x', 'y', 'z' or 'X', 'Y', 'Z') to create channel range array (Default is 'z')

    Returns
    -------
    np.array
        An array of channel values for given wavelength
    """
    print('DO NOT USE THIS')
    data_headers = datacontainer.get_data_header()
    channel_range = channel_array(datacube, channels)
    channel = np.array(channel_range) - data_headers['CRVAL3'] / data_headers['CD3_3']
    return np.array(channel, float)


def collapse_cube(datacube,
                  min_lambda=None,
                  max_lambda=None,
                  mask_z=None,
                  to_flux=True,
                  flux_val=1.25):
    """ Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    If mask_z is specified, the function will remove all channels masked as 1.

    Parameters
    ----------
    datacube : np.array
        3D data cube
    min_lambda : int, optional
        Minimum wavelength to collapse file
    max_lambda : int, optional
        Maximum wavelength to collapse file
    mask_z : np.array, optional
        range of z-axis values to mask when collapsing
    to_flux : boolean, optional
        converts collapsed data to flux values - erg/s/cm**2 (Default is True)
    flux_val : float, optional
        value for flux conversion (Default is 1.25 Angs.)

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

    # Sums values between specifications, ignoring NaNs
    if to_flux:
        col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :] * flux_val, axis=0)
    else:
        col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)

    del datacopy
    del z_max, y_max, x_max
    return col_cube


def collapse_cube_mask(datacontainer,
                       mask_xyz,
                       wave_scale=1.25,
                       statcube=None,
                       min_lambda=None,
                       max_lambda=None,
                       to_flux=True):
    """ Given a Cube, the macro collapses it along the z-axis between min_lambda and
    max_lambda. If maskXYZ is given, only voxels marked as 1 (or True) are considered.
    If to_flux is 'True' the units of the image are converted to erg/s/cm**2.
    Parameters
    ----------
    datacontainer : np.array, IFU cube
        data in a 3D array or initialized IFU cube
    mask_xyz : np.array
        when 1 (or True) this voxel will be considered
    wave_scale : float, optional
        if to_flux is True, this converts wavelength values
        if to_flux is False this defaults to 1. (default is 1.25)
    statcube : np.array, optional
        variance in a 3D array if datacontainer is also 3D array (Default is None)
    min_lambda : int
        min channel to create collapsed image
    max_lambda : int
        max channel to create collapsed image
    to_flux : bool
        it set to True, units are converted to erg/s/cm**2

    Returns
    -------
    collapsed_data, collapsed_stat: np.arrays
        Collapsed data and variance images
    """
    # Converting units if requested
    if to_flux:
        scale_factor = wave_scale
        print("collapse_cube_mask: Output converted to erg/s/cm**2.")
        print("                    (i.e. multiplied by {} Ang.)".format(scale_factor))
    else:
        scale_factor = 1.

    if statcube is None:
        tmp_datacube, tmp_statcube = datacontainer.get_data_stat()
    else:
        tmp_datacube = np.copy(datacontainer)
        tmp_statcube = np.copy(statcube)

    z_max, y_max, x_max = np.shape(tmp_datacube)

    if max_lambda is None or max_lambda > z_max:
        max_lambda = int(z_max)
        print("collapse_cube_mask : Exceeded / unspecified wavelength in data cube. "
              "Max value is set to {}".format(int(z_max)))
    if min_lambda is None or min_lambda < 0:
        min_lambda = 0
        print("collapse_cube_mask : Invalid / unspecified minimum wavelength. Min value is set to 0")

    # Masking channels in data (and in stat if present)
    tmp_datacube[(mask_xyz < 1)] = np.nan
    tmp_statcube[(mask_xyz < 1)] = np.nan

    # Collapsing data (and stat if present)
    collapsed_data = np.nansum(tmp_datacube[min_lambda:max_lambda, :, :] * scale_factor, axis=0)
    collapsed_stat = np.nansum(tmp_statcube[min_lambda:max_lambda, :, :] * scale_factor * scale_factor, axis=0)

    print("collapse_cube_mask: Images produced")

    # preserve memory
    del tmp_datacube
    del tmp_statcube
    del scale_factor

    return collapsed_data, collapsed_stat


def collapse_mean_cube(datacube,
                       statcube,
                       min_lambda=None,
                       max_lambda=None,
                       mask_z=None,
                       to_flux=True,
                       flux_val=1.25):
    """Given 3D arrays of data and variance, this function collapses it along the
    z-axis between min_lambda and max_lambda. If mask_z is given, channels masked
    as 1 (or True) are removed. The macro uses the stat information to perform a
    weighted mean along the velocity axis. In other words, each spaxel of the resulting
    image will be the weighted mean spectrum of that spaxels along the wavelengths.

    Parameters
    ----------
    datacube : np.array
        3D data file
    statcube : np.array
        3D variance file
    min_lambda : int
        minimum wavelength to collapse cube
    max_lambda : int
        maximum wavelength to collapse cube
    mask_z : np.array
        if specified, channels masked as 1 will be removed when collapsed
    to_flux : boolean, optional
        converts collapsed data to flux values - erg/s/cm**2 (Default is True)
    flux_val : float, optional
        value for flux conversion (Default is 1.25 Angs.)

    Returns
    -------
    np.arrays
        2D collapsed and averaged data and variance cubes
    """

    temp_collapsed_data = collapse_cube(datacube, min_lambda,
                                        max_lambda, mask_z=mask_z,
                                        to_flux=to_flux, flux_val=flux_val)
    temp_collapsed_stat = collapse_cube(statcube, min_lambda,
                                        max_lambda, mask_z=mask_z,
                                        to_flux=to_flux, flux_val=flux_val)

    temp_collapsed_stat = 1. / temp_collapsed_stat
    bad_pix = np.isnan(temp_collapsed_data) | np.isinf(temp_collapsed_stat)
    # to deal with np.nan a weight of (almost) zero is given to bad_pix
    temp_collapsed_data[bad_pix] = 0.
    temp_collapsed_stat[bad_pix] = np.nanmin(temp_collapsed_stat) / 1000.
    collapsed_data, collapsed_weights = np.average(temp_collapsed_data,  # Todo depreciated function with code
                                                   weights=temp_collapsed_stat,
                                                   axis=0,
                                                   returned=True)
    collapsed_stat = 1. / collapsed_weights

    # preserve memory
    del temp_collapsed_data
    del temp_collapsed_stat
    del collapsed_weights

    return collapsed_data, collapsed_stat


def collapse_container(datacontainer,
                       min_lambda=None,
                       max_lambda=None,
                       mask_z=None,
                       to_flux=True,
                       flux_val=1.25,
                       var_thresh=5.):
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
    to_flux : boolean, optional
        converts collapsed data to flux values - erg/s/cm**2 (Default is True)
    flux_val : float, optional
        value for flux conversion (Default is 1.25 Angs.)
    var_thresh : int, float, optional
        if no variance in container, this value determines standard deviation threshold
         to create variance cube from data (Default is 5.)

    Returns
    -------
    np.array
        Condensed 2D array of 3D file.

    """
    # pulls information from IFU object
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

    if to_flux:
        col_datacube = np.nansum(datacopy[min_lambda:max_lambda, :, :] * flux_val, axis=0)
    else:
        col_datacube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)

    if statcopy is not None:
        if mask_z is not None:
            statcopy[mask_z, :, :] = np.nan
        if to_flux:
            col_statcube = np.nansum(statcopy[min_lambda:max_lambda, :, :] * flux_val, axis=0)
        else:
            col_statcube = np.nansum(statcopy[min_lambda:max_lambda, :, :], axis=0)

    else:
        # generation of variance stemming from dataset
        med_cube = np.nanmedian(col_datacube)
        std_cube = np.nanstd(col_datacube - med_cube)
        tmp_col_statcube = \
            np.zeros_like(col_datacube) + np.nanvar(col_datacube[(col_datacube - med_cube) < (var_thresh * std_cube)])
        if mask_z is not None:
            tmp_col_statcube[mask_z, :, :] = np.nan
        if to_flux:
            col_statcube = np.nansum(tmp_col_statcube[min_lambda:max_lambda, :, :] * flux_val, axis=0)
        else:
            col_statcube = np.nansum(tmp_col_statcube[min_lambda:max_lambda, :, :], axis=0)

        del tmp_col_statcube
        del med_cube
        del std_cube
    del datacopy
    del statcopy
    del z_max, y_max, x_max

    return col_datacube, col_statcube


def collapse_mean_container(datacontainer,
                            min_lambda=None,
                            max_lambda=None,
                            mask_z=None,
                            threshold=5.):
    """Given an IFUcube, collapse it along the z-axis between min_lambda and
    max_lambda. If mask_z is given, channels masked as 1 (or True) are removed. The
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
        deviation for the creation of this variance (Default is 5.)

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

    # preserve memory
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
        Semi-major axis (Default is 10)
    semi_min : int, float, optional
        Semi-minor axis (Default is 6)
    theta : int, float, optional
        Angle for ellipse rotation around object (Default is 0)

    Returns
    -------
    np.array
        Mask of 2D array of with user defined stellar objects
        denoted as 1 with all other elements 0
    """

    mask_array = np.zeros_like(data_flat, float)
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
        return mask_array  # returned mask of zeros
    # checks data type of x_position to determine multitude of objects

    if isinstance(x_position, int | float):
        theta_rad = (theta * np.pi) / 180.  # converts angle degrees to radians
        object_ellipse = EllipticalAperture(object_position, semi_maj, semi_min, theta_rad)
        ellipse_mask = object_ellipse.to_mask(method="center").to_image(shape=(x_mask, y_mask))
        mask_array += ellipse_mask
    else:
        # accounts for single input of angle and semi-diameter values
        # creates full arrays of the specified value to apply to all objects
        print('location: multiple sources identified')

        if isinstance(theta, int | float):
            theta = np.zeros_like(x_position)
        if isinstance(semi_min, int | float):
            semi_min = np.full_like(x_position, semi_min)
        if isinstance(semi_maj, int | float):
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


def annular_mask(data_flat,
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
        inner semi-major axis in pixel (Default is 5.)
    semi_min : float, optional
        inner semi-minor axis in pixel (Default is 5.)
    delta_maj : float, optional
        width of the semi-major axis in pixel (Default is 2.)
    delta_min : float, optional
        width of the semi-minor axis in pixel (Default is 2.)
    theta : float, optional
        angle wrt the x-axis in degrees (Default is 0.)

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


def small_cube(datacontainer,
               min_lambda=None,
               max_lambda=None):
    """Given an IFUcube, this function collapses along min_lambda
    and max_lambda. It also updates the wavelength information to
    be conformed to the new size. Note that min_lambda and max_lambda
    can be given both as wavelength and as channels. If the input value is
    <3000. it will be assumed to be channel number, otherwise
    wavelength in Angstrom will be considered.

    Parameters
    ----------
    datacontainer : IFUcube object
        data initialized in cubeClass.py
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
    # assignment of variables to all information from IFU cube
    primary_headers = datacontainer.get_primary()
    datacopy, statcopy = datacontainer.get_data_stat()
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
        print("small_cube: min_lambda set to 0")
        min_lambda = 0
    if max_lambda is None:
        print("small_cube: max_lambda set to {}".format(int(z_max)))
        max_lambda = int(z_max)
    # If input values are larger than 3000., the macro converts from
    # wavelength in ang. to channel number.
    if min_lambda > 3000.:
        print("small_cube: Converting min wavelength in Ang. to channel number")
        min_lambda = int((np.array(min_lambda - data_headers['CRVAL3']) / data_headers['CD3_3']))
    else:
        min_lambda = int(min_lambda)
    if max_lambda > 3000.:
        print("small_cube: Converting Max wavelength in Ang. to channel number")
        max_lambda = int((np.array(max_lambda - data_headers['CRVAL3']) / data_headers['CD3_3']))
    else:
        max_lambda = int(max_lambda)
    # Check for upper and lower limits
    if min_lambda < 0:
        print("small_cube: Negative value for min_lambda set to 0")
        min_lambda = int(0)
    if max_lambda > (z_max + 1):
        print("small_cube: max_lambda is outside the cube size. Set to {}".format(int(z_max)))
        max_lambda = int(z_max)
    # updates CRVAL3 of new data cube
    small_cube_CRVAL3 = float(np.array(data_headers['CRVAL3'] + (np.array(min_lambda) * data_headers['CD3_3'])))
    print("small_cube: Creating smaller cube")
    print("           The old pivot wavelength was {}".format(data_headers['CRVAL3']))
    print("           The new pivot wavelength is {}".format(small_cube_CRVAL3))
    # Header
    s_primary_headers = copy.deepcopy(primary_headers)
    # DATA
    s_data_headers = copy.deepcopy(data_headers)
    s_data_headers['CRVAL3'] = small_cube_CRVAL3
    s_datacopy = np.copy(datacopy[min_lambda:max_lambda, :, :])
    print('small_cube: New data and variance shape ',s_datacopy.shape)
    # STAT
    s_stat_headers = copy.deepcopy(stat_headers)
    s_stat_headers['CRVAL3'] = small_cube_CRVAL3
    s_statcopy = np.copy(statcopy[min_lambda:max_lambda, :, :])

    return s_primary_headers, s_data_headers, s_datacopy, s_stat_headers, s_statcopy


def small_IFU(datacontainer,
              image,
              min_lambda=None,
              max_lambda=None):
    """Creates a smaller IFU datacube utilizing function of small cube. Benefit of function is to return a full IFUcube
    object with corrected shrunken parameters. Helps to save memory -> computation time when dealing with large
    amounts of data.
    Parameters
    ----------
    datacontainer : IFUcube Object
        dataset initialized in cubeClass
    image : str
        pathway to image used for datacontainer
    min_lambda : int, optional
        minimum wavelength (z value) to create smaller data set from
    max_lambda : int, optional
        maximum wavelength (z value) to create smaller data set from
    Returns
    -------
    SmallIFU : IFUcube Object
        shrunken and parameter corrected datacube
    """
    s_primary_headers, s_data_headers, s_datacopy, s_stat_headers, s_statcopy = small_cube(datacontainer,
                                                                                           min_lambda,
                                                                                           max_lambda)

    primary = fits.PrimaryHDU()
    data = fits.ImageHDU(s_datacopy, s_data_headers, name='DATA')
    stat = fits.ImageHDU(s_statcopy, s_stat_headers, name='STAT')
    smallIFU = cubeClass.IfuCube(image,
                                 primary=primary,
                                 data=data,
                                 stat=stat)
    return smallIFU


def dust_correction(datacontainer, statcube=None):
    """Function queries the IRSA dust map for E(B-V) value and
    returns a reddening array. Works along z-axis of datacube
    http://irsa.ipac.caltech.edu/applications/DUST/
    The query return E(B_V) from SFD (1998). This will be converted
    to the S&F (2011) one using:
    E(B-V)S&F =  0.86 * E(B-V)SFD

    Parameters
    ----------
    datacontainer : IFU object, np.array
        3D .fits file array / IFU cube object
    statcube : np.array, optional
        3D variance array, optional if passing an IFU cube (Default is None)
    Returns
    -------
    reddata, redstat : np.array
        galactic dust corrected 3D arrays for data and variance
    """
    if statcube is None:
        reddata, redstat = datacontainer.get_data_stat()
    else:
        reddata = datacontainer
        redstat = statcube
    channel_range = channel_array(reddata, 'z')  # creates channel along z-axis

    headers = datacontainer.get_primary()
    ra = headers['RA']
    dec = headers['DEC']
    wavecube = convert_to_wave(datacontainer, reddata)  # creates wavelength value cube along z-axis

    coordinates = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='fk5')
    try:
        dust_image = IrsaDust.get_images(coordinates, radius=5. * u.deg, image_type='ebv', timeout=60)[0]
    except TimeoutError:
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

    return reddata, redstat


def sb_profile(datacontainer,
               x_pos,
               y_pos,
               rad_min=2.,
               rad_max=30.,
               r_bin=5.,
               speed=500,
               rest_lambda=1215.67,
               pixel_scale=0.2,
               log_bin=True,
               z=None,
               bad_pixel_mask=None,
               bg_correct=True,
               output='Object',
               debug=False,
               show_debug=False):
    """Performing quick circular aperture photometry on an image
    and extracting the surface brightness profile (including errors).
    The average background will be estimated from the outer
    annulus.

    Parameters
    ----------
    datacontainer : IFUcube Object
        dataset initialized in cubeClass
    x_pos : float
        x-location of the source in pixel
    y_pos : float
        y-location of the source in pixel
    rad_min : float, optional
        radius in pixel where to start to estimate the SB profile (Default is 2.)
    rad_max : float, optional
        radius in pixel where to stop the estimate of the SB profile (Default is 30.)
    r_bin : float, optional
        linear step in pixel to go from rad_min to rad_max (Default is 5.)
    speed : float, optional
        estimated gas velocity to generate wavelength range to search (Default 500 km/s)
    rest_lambda : float, optional
        rest frame wavelength for emission of interest
        (Default is LyA (1215.67 Ang))
    pixel_scale : int, float, optional
        instrument pixel scale correction (Default 0.2)
    log_bin : bool
        if True, the bin will be equally spaced in log10 space
        in this case, r_bin is considered as the size of the first bin
        (Default is True)
    z : float
        redshift value used to accurately calculate surface brightness
    bad_pixel_mask : np.array
        2D mask marking spatial pixels to be removed from the
        profile extraction
    bg_correct : bool
        If True, the average value of the flux from the outer
        bin will be subtracted from the data
        (Default is True)
    output : string, optional
        root file name for output (Default is 'Object')
    debug, show_debug : boolean, optional
        runs debug sequence to display output of function (Default is False)

    Returns
    -------
    radius, sb, var_sb : np.arrays
        surface brightness profile and errors. The radius and
        surface brightness are converted to arcseconds. (i.e.
        are not in pixel units).
    rad_diff : np.array
        length of a bin in radius in arcseconds.
    background_sb : np.array
        1-sigma error for the background surface brightness
        profile
    sigma_background : float
        average std for the outer ring
    """

    print("sb_profile: SB profile estimation")

    # Removing NaNs and bad pixels
    lambda_obs = (1. + z) * rest_lambda  # in Ang.
    c = 299792.458  # in km/s
    sb_min = lambda_obs * (1. - (speed / c))
    sb_max = lambda_obs * (1. + (speed / c))
    data_headers = datacontainer.get_data_header()
    # converting redshifted wavelength of sb limits to channel numbers
    channel_min = int((sb_min - data_headers['CRVAL3']) / data_headers['CD3_3'])
    channel_max = int((sb_max - data_headers['CRVAL3']) / data_headers['CD3_3'])
    print('sb_profile: collapsing cube between {} - {}'.format(channel_min, channel_max))
    tmp_data, tmp_stat = collapse_container(datacontainer,
                                            min_lambda=channel_min,
                                            max_lambda=channel_max)


    # make copy of input data image for debug sequence.
    data_copy = np.copy(tmp_data)

    if bad_pixel_mask is not None:
        print("sb_profile: removing bad pixels")
        tmp_data[(bad_pixel_mask > 0)] = np.nan
        tmp_stat[(bad_pixel_mask > 0)] = np.nan

    img_good_pix = np.ones_like(tmp_data, dtype=float)
    # pixels in image that are undefined or infinite become 0s
    img_good_pix[(~np.isfinite(tmp_stat))] = 0.
    img_good_pix[(~np.isfinite(tmp_data))] = 0.
    tmp_stat[(~np.isfinite(tmp_stat))] = 0.
    tmp_data[(~np.isfinite(tmp_data))] = 0.

    # setting up arrays
    if log_bin:
        start_rad = np.log10(rad_min)
        bin_rad = np.log10(rad_min + r_bin) - np.log10(rad_min)
        end_rad = np.log10(rad_max)
        log_min_rad = np.arange(start_rad, end_rad + bin_rad, bin_rad)
        log_max_rad = log_min_rad + bin_rad
        min_rad = log_min_rad ** 10
        max_rad = log_max_rad ** 10
        radius = (min_rad + 0.5 * (max_rad - min_rad)) * pixel_scale
    else:
        min_rad = np.arange(rad_min, rad_max + r_bin, r_bin)
        max_rad = min_rad + r_bin
        radius = (min_rad + 0.5 * r_bin) * pixel_scale
    rad_diff = (max_rad - min_rad) * pixel_scale

    print("sb_profile: {:0.0f} bins between R={:0.2f} and {:0.2f} arcsec".format(np.size(radius),
                                                                                 min(min_rad) * 0.2,
                                                                                 max(max_rad) * 0.2))

    print("sb_profile: Checking for background stats")

    # set up boundary mask of 0 outside, 1 inside.
    outer_ann_mask = annular_mask(tmp_data,
                                  x_pos=x_pos,
                                  y_pos=y_pos,
                                  semi_maj=min_rad[-1],
                                  semi_min=min_rad[-1],
                                  delta_maj=max_rad[-1] - min_rad[-1] - 2.,
                                  delta_min=max_rad[-1] - min_rad[-1] - 2.,
                                  theta=0.)

    avg_data_median = np.nanmedian(tmp_data[(outer_ann_mask > 0) & (img_good_pix > 0.)])
    avg_data_var = np.nanvar(tmp_data[(outer_ann_mask > 0) & (img_good_pix > 0.)])
    avg_stat_var = np.nanmedian(tmp_stat[(outer_ann_mask > 0) & (img_good_pix > 0.)])
    print("                the variance of the b/g DATA is: {:0.3f}".format(avg_data_var))
    print("                the average variance of the b/g STAT is: {:0.3f}".format(avg_stat_var))
    print("                the average of the b/g DATA is: {:0.3f}".format(avg_data_median))
    var_ratio = avg_data_var / avg_stat_var
    if ((var_ratio < 0.9) | (var_ratio > 1.1)) & (var_ratio > 0.):
        print("sb_profile: Extra correction for STAT image of a factor {:0.3f}".format(var_ratio))
        tmp_stat = tmp_stat * var_ratio
    if bg_correct:
        print("sb_profile: Extra correction for b/g counts applied : -{:0.3f} counts".format(avg_data_median))
        tmp_data[(img_good_pix > 0)] = tmp_data[(img_good_pix > 0)] - avg_data_median
    del var_ratio
    del outer_ann_mask

    sb = np.zeros_like(radius, dtype=float)
    var_sb = np.copy(sb)
    background_sb = np.copy(sb)

    obj_pos = [x_pos, y_pos]

    # Estimating background values
    obj_ann = CircularAnnulus(obj_pos, r_in=min_rad[-1], r_out=max_rad[-1])
    bg_aperture_var = aperture_photometry(tmp_stat, obj_ann)
    bg_aperture_pix = aperture_photometry(img_good_pix, obj_ann)
    pix_var_bg = float(np.array(bg_aperture_var['aperture_sum'][0]) / np.array(bg_aperture_pix['aperture_sum'][0]))
    for rad_index in np.arange(0, np.size(min_rad)):
        obj_ann = CircularAnnulus(obj_pos, r_in=min_rad[rad_index], r_out=max_rad[rad_index])
        # Flux
        flux_aperture = aperture_photometry(tmp_data, obj_ann)
        # Variance
        variance_aperture = aperture_photometry(tmp_stat, obj_ann)
        # Pixels number
        pixel_aperture = aperture_photometry(img_good_pix, obj_ann)

        sb[rad_index] = float(np.array(flux_aperture['aperture_sum'][0]) / (
                pixel_scale * pixel_scale * np.array(pixel_aperture['aperture_sum'][0])))

        var_sb[rad_index] = float(np.sqrt(np.array(variance_aperture['aperture_sum'][0])) / (
                pixel_scale * pixel_scale * np.array(pixel_aperture['aperture_sum'][0])))
        background_sb[rad_index] = float(np.sqrt(pix_var_bg * np.array(pixel_aperture['aperture_sum'][0])) / (
                pixel_scale * pixel_scale * np.array(pixel_aperture['aperture_sum'][0])))
    sigma_background = np.sqrt(pix_var_bg) / (pixel_scale * pixel_scale)

    if debug:

        print("sb_profile: Creating debug image")

        plt.figure(1, figsize=(12, 6))

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_mask = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        ax_image.imshow(tmp_data / (pixel_scale * pixel_scale),
                        cmap="Greys", origin="lower",
                        vmin=-1. * sigma_background,
                        vmax=+3. * sigma_background)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_xlim(left=x_pos - max_rad[-1], right=x_pos + max_rad[-1])
        ax_image.set_ylim(bottom=y_pos - max_rad[-1], top=y_pos + max_rad[-1])
        ax_image.set_title(r"Data")
        for idxRadius in np.arange(0, np.size(radius)):
            min_artist = Ellipse(xy=(x_pos, y_pos),
                                 width=2. * min_rad[idxRadius],
                                 height=2. * min_rad[idxRadius],
                                 angle=0.)
            min_artist.set_facecolor('none')
            min_artist.set_edgecolor("red")
            min_artist.set_alpha(0.5)
            max_artist = Ellipse(xy=(x_pos, y_pos),
                                 width=2. * max_rad[idxRadius],
                                 height=2. * max_rad[idxRadius],
                                 angle=0.)
            max_artist.set_facecolor("none")
            max_artist.set_edgecolor("blue")
            max_artist.set_alpha(0.5)
            ax_image.add_artist(min_artist)
            ax_image.add_artist(max_artist)

        # Plotting mask image
        ax_mask.imshow(img_good_pix - 1,
                       cmap="Greys_r", origin="lower",
                       vmin=-1,
                       vmax=0)
        ax_mask.set_xlabel(r"X [Pixels]", size=30)
        ax_mask.set_ylabel(r"Y [Pixels]", size=30)
        ax_mask.set_xlim(left=x_pos - max_rad[-1], right=x_pos + max_rad[-1])
        ax_mask.set_ylim(bottom=y_pos - max_rad[-1], top=y_pos + max_rad[-1])
        ax_mask.set_title(r"Excluded Pixels Mask")

        for idxRadius in np.arange(0, np.size(radius)):
            min_artist = Ellipse(xy=(x_pos, y_pos),
                                 width=2. * min_rad[idxRadius],
                                 height=2. * min_rad[idxRadius],
                                 angle=0.)
            min_artist.set_facecolor("none")
            min_artist.set_edgecolor("red")
            min_artist.set_alpha(0.5)
            max_artist = Ellipse(xy=(x_pos, y_pos),
                                 width=2. * max_rad[idxRadius],
                                 height=2. * max_rad[idxRadius],
                                 angle=0.)
            max_artist.set_facecolor("none")
            max_artist.set_edgecolor("blue")
            max_artist.set_alpha(0.5)
            ax_mask.add_artist(min_artist)
            ax_mask.add_artist(max_artist)

        print("sb_profile: debug image saved in {}_SBProfile.pdf".format(output))
        plt.tight_layout()
        plt.savefig(output + "_SBProfile.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if show_debug:
            plt.show()
        plt.close()

    return radius, sb, var_sb, rad_diff, background_sb, sigma_background

def quick_ap_photometry(datacopy,
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
        radius where to perform the aperture photometry (Default is 2.)
    inner_rad : np.array, optional
        inner radius of the background region in pixel (Default is 10.)
    outer_rad : np.array, optional
        outer radius of the background region in pixel (Default is 15.)

    Returns
    -------
    obj_flux, obj_err_flux, ave_flux : np.array
        flux of object, sigma of flux, and average flux values
    """

    print("quick_ap_photometry: Performing aperture photometry")
    if statcopy is None:
        print('quick_ap_photometry: No variance cube argument')
        statcopy = np.ones_like(datacopy)
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


def quick_ap_photometry_no_bg(datacopy,
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
        radius where to perform the aperture photometry (Default is 2.)

    Returns
    -------
    flux_obj, err_flux_obj : np.arrays
        fluxes and errors of the sources derived from datacopy
        and statcopy
    """

    print("quick_ap_photometry_no_bg: Performing aperture photometry")

    # creates np.arrays of position values regardless of quantity
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

    # filling arrays
    print("quick_ap_photometry_no_bg: Aperture photometry on source:")
    for index in range(0, np.size(x_object)):
        print("                      {:03d} -> ({:03.2f},{:03.2f})".format(index, float(x_object[index]),
                                                                           float(y_object[index])))
        object_pos = (float(x_object[index]), float(y_object[index]))
        circ_obj = CircularAperture(object_pos, r=float(r_aperture[index]))
        ap_phot = aperture_photometry(datacopy, circ_obj)
        var_ap_phot = aperture_photometry(statcopy, circ_obj)
        flux_obj[index] = ap_phot['aperture_sum'][0]
        err_flux_obj[index] = var_ap_phot['aperture_sum'][0]**0.5

    # Deleting temporary arrays to clear up memory
    del r_aperture
    del object_pos
    del circ_obj
    del ap_phot
    del var_ap_phot

    return flux_obj, err_flux_obj


def q_spectrum(datacontainer,
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
        Otherwise, pulls this array from IFUcube object (Default is None)
    radius_pos: np.array, optional
        radius where to perform the aperture photometry (Default is 2.)
    inner_rad: np.array, optional
        inner radius of the background region in pixel (Default is 10.)
    outer_rad: np.array, optional
        outer radius of the background region in pixel (Default is 15.)
    void_mask : np.array, optional
        mask of possible contaminants (1->the pixel will be
        removed from the photometry) (Default is None)

    Returns
    -------
    spec_ap_phot : np.array
        spectra of selected object
    spec_var_ap_phot : np.array
        sigma of spectra
    spec_flux_bg : np.array
        background spectral flux
    """

    # allows assignment of an IFUcube object or two 3D arrays
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = datacontainer
        statcopy = np.copy(statcube)

    if void_mask is None:
        void_mask = np.zeros_like(datacopy[0, :, :])
    else:
        print("q_spectrum: Using void_mask")

    # creating empty lists
    spec_ap_phot = []
    spec_var_ap_phot = []
    spec_flux_bg = []
    object_pos = [x_pos, y_pos]

    circ_obj = CircularAperture(object_pos, r=radius_pos)
    annu_obj = CircularAnnulus(object_pos, r_in=inner_rad, r_out=outer_rad)
    z_max, y_max, x_max = datacopy.shape

    for channel in np.arange(0, z_max, 1):
        # Total flux
        temp_data = (datacopy[channel, :, :])
        ap_phot = aperture_photometry(temp_data, circ_obj)
        # Background masking bad values
        temp_data_bg = (datacopy[channel, :, :])
        bad_bg = np.min([np.nanmin(temp_data_bg), -99999.])
        temp_data_bg[(void_mask == 1)] = bad_bg
        temp_data_bg[~np.isfinite(temp_data_bg)] = bad_bg
        mask_bg = annu_obj.to_mask(method='center')
        data_bg = mask_bg.multiply(temp_data_bg)
        data_bg[(np.array(mask_bg) == 0)] = bad_bg
        data_bg1d = data_bg[data_bg > bad_bg].flatten()
        mean_bg, median_bg, sigma_bg = sigma_clipped_stats(data_bg1d)
        # Error
        temp_stat = (statcopy[channel, :, :])
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


def q_spectrum_no_bg(datacontainer,
                     x_pos,
                     y_pos,
                     statcube=None,
                     radius_pos=2.):
    """Performing quick spectrum extraction from a circular aperture on a cube
    without calculating background spectral flux

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
        Otherwise, pulls this array from IFUcube object (Default is None)
    radius_pos : float
        radius where to perform the aperture photometry

    Returns
    -------
    spec_ap_phot : np.array
        spectra of selected object
    spec_var_ap_phot : np.array
        sigma of spectra
    """

    print("q_spectrum_no_bg: Extracting spectrum from the cube")
    # allows IFUcube object assignment or two 3D arrays
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = np.copy(datacontainer)
        statcopy = np.copy(statcube)

    # creating empty lists
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


def q_spectrum_no_bg_mask(datacontainer,
                          mask_xy, statcube=None):
    """Performing quick spectrum extraction from an aperture given by a 2D mask
    from a cube.

    Parameters
    ----------
    datacontainer : IFUcube object, np.array
        data initialized in cubeClass.py or 3D array of data
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

    # allows IFUcube object assignment or two 3D arrays
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = np.copy(datacontainer)
        statcopy = np.copy(statcube)

    print("q_spectrum_no_bg_mask: Extracting spectrum from the cube")
    z_max, y_max, x_max = np.shape(datacopy)
    for channel in np.arange(0, z_max, 1):
        # masking arrays
        datacopy[channel, :, :][(mask_xy < 1)] = 0
        statcopy[channel, :, :][(mask_xy < 1)] = 0

    flux_obj = np.nansum(datacopy, axis=(1, 2))
    err_flux_obj = (np.nansum(statcopy, axis=(1, 2)) ** 0.5)

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


def stat_fullcube(datacube,
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
        the cube will be masked (Default is None)

    Returns
    -------
    cube_average, cube_median, cube_std : float
        average, median, and standard deviation of
        the cube.
    """

    print("stat_fullcube: statistic on the cube")
    datacopy = np.copy(datacube)

    cube_average = np.nanmean(datacopy)
    cube_median = np.nanmedian(datacopy)
    cube_std = np.nanstd(datacopy)

    if n_sigma_extreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extreme_mask = np.zeros_like(datacopy, int)
        datacopy[~np.isfinite(datacopy)] = cube_average + (3. * n_sigma_extreme * cube_std)
        extreme_mask[np.abs((datacopy - cube_average) / cube_std) > n_sigma_extreme] = 1
        cube_average = np.nanmean(datacopy[(extreme_mask == 0)])
        cube_median = np.nanmedian(datacopy[(extreme_mask == 0)])
        cube_std = np.nanstd(datacopy[(extreme_mask == 0)])
        del extreme_mask

    print("stat_fullcube: average = {:+0.3f}".format(cube_average))
    print("              median  = {:+0.3f}".format(cube_median))
    print("              sigma   = {:+0.3f}".format(cube_std))

    # preserve memory
    del datacopy

    return cube_average, cube_median, cube_std


def stat_fullcube_z(datacube,
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
        the cube will be masked (Default is None)

    Returns
    -------
    cube_avg_z, cube_med_z, cube_std_z : np.arrays
        average, median, and standard deviation of
        the cube along the spectral axis.
    """

    print("stat_fullcube_z: statistic on the cube")
    datacopy = np.copy(datacube)

    cube_avg_z = np.nanmean(datacopy, axis=(1, 2))
    cube_med_z = np.nanmedian(datacopy, axis=(1, 2))
    cube_std_z = np.nanstd(datacopy, axis=(1, 2))

    if n_sigma_extreme is not None:
        # Workaround to avoid problems generated by using np.abs and NaNs
        extreme_mask = np.zeros_like(datacopy, int)
        z_max, y_max, x_max = np.shape(datacopy)
        for channel in np.arange(0, z_max):
            datacopy[channel, :, :][~np.isfinite(datacopy[channel, :, :])] = cube_avg_z[channel] + (
                    3. * n_sigma_extreme * cube_std_z[channel])
            extreme_mask[channel, :, :][np.abs(
                (datacopy[channel, :, :] - cube_avg_z[channel]) / cube_std_z[channel]) > n_sigma_extreme] = 1
        datacopy[(extreme_mask == 1)] = np.nan
        cube_avg_z = np.nanmean(datacopy, axis=(1, 2))
        cube_med_z = np.nanmedian(datacopy, axis=(1, 2))
        cube_std_z = np.nanstd(datacopy, axis=(1, 2))
        del extreme_mask

    # Cleaning up memory
    del datacopy

    return cube_avg_z, cube_med_z, cube_std_z


def pixel_dist(z_pos1, y_pos1, x_pos1,
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

    z_dist = (z_pos1 - z_pos2)**2
    y_dist = (y_pos1 - y_pos2)**2
    x_dist = (x_pos1 - x_pos2)**2

    dist = np.sqrt(z_dist + y_dist + x_dist)

    return dist


def object_coord(datacontainer, ra, dec, radius, debug=True):
    """Reads in IFUcube and user input of an object in RA and DEC as well as a search radius.
    RA and DEC specify where to extend radius to search for pixel value of center of object.
    Utilizes Photutils centroid_sources and header data to convert RA and DEC into pixel values.

    Parameters
    ----------
    datacontainer : IFUcube Object
        Data file initialized in cubeClass.py
    ra : float
        Right ascension of object in degrees
    dec : float
        Declination of object in degrees
    radius : int, float
        Pixel radius value to search for center of specified object
    debug : bool, optional
        If true, plots image with circle drawn over object location. (Default is False)
    Returns
    -------
    float
        x_pos X pixel coordinate of object in datacube
        y_pos Y pixel coordinate of object in datacube
    """

    headers = datacontainer.get_data_header()
    datacopy = datacontainer.get_data()
    image_data = np.nansum(datacopy[:, :, :], axis=0)  # creates 2D image from dataset and deletes copy to save memory
    del datacopy

    # utilizes header information for conversions
    ra_ref = headers['CRVAL1']
    ra_conversion = headers['CD1_1']
    x_ref = headers['CRPIX1']

    dec_ref = headers['CRVAL2']
    dec_conversion = headers['CD2_2']
    y_ref = headers['CRPIX2']

    ra_dif = ra - ra_ref
    dec_dif = dec - dec_ref
    x_pix = np.array((ra_dif / ra_conversion) + x_ref)
    y_pix = np.array((dec_dif / dec_conversion) + y_ref)

    # isolates position fof the center of the object
    x_pos, y_pos = centroids.centroid_sources(image_data, x_pix, y_pix,
                                              box_size=radius)

    if debug:
        fig = plt.figure()
        plt.imshow(image_data, origin='lower', vmin=np.nanmedian(image_data),
                   vmax=np.nanmedian(image_data)+3*np.nanstd(image_data))
        plt.xlim(x_pos - 40, x_pos + 40)
        plt.ylim(y_pos - 40, y_pos + 40)
        ax = plt.gca()
        circ = patches.Circle((x_pos, y_pos), 5, fc=None, fill=None)
        ax.add_patch(circ)

    # converts back to float for returned coordinates
    x_pos = float(x_pos)
    y_pos = float(y_pos)
    return x_pos, y_pos


def measure_line(wave,
                 flux,
                 error=None,
                 min_lambda=None,
                 max_lambda=None,
                 emission_only=True):
    """Given a spectrum containing an emission line
    the macro estimates the central wavelength, the
    FWHM (full width at half maximum), and the integral of the flux.

    Parameters
    ----------
    wave, flux : np.arrays
        wavelength and flux
    error : np.array
        if given, the errors on the measurement will
        also be returned
    emission_only : bool
        if True, negative fluxes will be ignored in the
        integral.
    min_lambda, max_lambda : float
        if given, the estimate will be done only on this
        wavelength range
    Returns
    -------
    average, average_err : floats
        average and the average error
    fwhm, fwhm_err : floats
        FWHM and the FWHM error
    integral, integral_err : floats
        integral and its error
    """
    print("measure_line: extracting information from the spectrum")
    wavelength = np.copy(wave)
    flux_density = np.copy(flux)
    good_waves = np.ones_like(wavelength, float)
    if error is None:
        err_flux_density = np.ones_like(flux_density, float)
    else:
        err_flux_density = np.copy(error)
    if emission_only:
        print("measure_line: Considering only positive flux")
        flux_density[(flux_density < 0.)] = 0.
    if min_lambda is not None:
        good_waves[wavelength < min_lambda] = 0.
        flux_density[wavelength < min_lambda] = 0.
        err_flux_density[wavelength < min_lambda] = 1.
    if max_lambda is not None:
        good_waves[wavelength > max_lambda] = 0.
        flux_density[wavelength > max_lambda] = 0.
        err_flux_density[wavelength > max_lambda] = 1.
    weights = 1. / error ** 2.
    # find average
    weight_sum = np.nansum(flux_density * weights)
    weight_sum_sqrd = np.nansum((flux_density * weights) ** 2.)
    unbias_factor = (1. - (weight_sum_sqrd / (weight_sum ** 2.)))
    average = np.nansum(wavelength * flux_density * weights) / weight_sum
    variance = np.nansum(flux_density * weights * ((wavelength - average) ** 2.)) / weight_sum
    unbias_variance = variance / unbias_factor
    average_err = np.sqrt(unbias_variance / np.nansum(good_waves))

    # Calculate Integral
    delta_wave = wavelength - np.roll(wavelength, 1)
    delta_wave[0] = delta_wave[1]
    print("measure_line: the length of one pixel is between {:0.3f} and {:0.3f} Ang.".format(np.nanmin(delta_wave),
                                                                                             np.nanmax(delta_wave)))
    integral = np.nansum(flux_density * delta_wave)
    if error is not None:
        integral_err = np.sqrt(np.nansum((err_flux_density[(good_waves > 0)] ** 2.)))
    else:
        integral_err = -1.

    # find FWHM
    # Removing low S/N pixels
    bad_sigma = np.where((err_flux_density - np.nanmedian(err_flux_density)) > 2. * np.nanstd(err_flux_density))

    # Smooth spectrum to avoid dependencies on few bright peaks
    smooth_flux_density = ndimage.gaussian_filter(flux_density, 5)
    flux_density_cleaner = np.copy(flux_density)
    flux_density_cleaner[bad_sigma] = smooth_flux_density[bad_sigma]
    smooth_flux_density = ndimage.gaussian_filter(flux_density_cleaner, 2)
    del flux_density_cleaner
    peak_flux = np.max(smooth_flux_density[(good_waves > 0)])
    where_peak = np.where(smooth_flux_density == peak_flux)
    error_at_peak = float(np.nanmin([err_flux_density[where_peak],
                                     np.nanmedian(err_flux_density[(good_waves > 0)])]))
    where_bright = np.where(smooth_flux_density >= 0.5 * peak_flux)
    where_bright_up = np.where(smooth_flux_density >= 0.5 * (peak_flux + error_at_peak))
    if np.size(where_bright_up) < 1:
        where_bright_up = np.where(smooth_flux_density >= 0.99 * peak_flux)
    where_bright_low = np.where(smooth_flux_density >= 0.5 * (peak_flux - error_at_peak))
    fwhm = (np.max(wavelength[where_bright]) - np.min(wavelength[where_bright]))
    fwhm_corrected = np.sqrt(fwhm**2-5**2)
    fwhm_up = np.max(wavelength[where_bright_up]) - np.min(wavelength[where_bright_up])
    fwhm_corrected_up = np.sqrt(fwhm_up**2-5**2)
    fwhm_low = np.max(wavelength[where_bright_low]) - np.min(wavelength[where_bright_low])
    fwhm_corrected_low = np.sqrt(fwhm_low**2-5**2)
    if error is not None:
        err_fwhm = np.max([np.abs(fwhm_corrected - fwhm_corrected_up),
                           np.abs(fwhm_corrected - fwhm_corrected_low),
                           np.nanmedian(delta_wave)]) / np.sqrt(2.)
    else:
        err_fwhm = -1.

    del wavelength, flux_density, err_flux_density, weights

    return average, average_err, fwhm_corrected, err_fwhm, integral, integral_err
