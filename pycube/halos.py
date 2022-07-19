""" Module to search extended emission in PSF subtracted MUSE cubes
"""
import numpy as np
import matplotlib.pyplot as plt
import gc
import astropy

from matplotlib import gridspec
from matplotlib.patches import Ellipse

from scipy import ndimage
from scipy.optimize import curve_fit

from astropy.convolution import convolve

from pycube.core import manip


cSpeed = 299792.458  # in km/s


def smoothChiCube(datacontainer,
                  min_lambda,
                  max_lambda,
                  statcube=None,
                  a_smooth=1.,
                  v_smooth=1.,
                  truncate=5.):
    """Given a (PSF subtracted) IFUcube, the macro smooth both
    the datacube and the statcube with a 3D gaussian Kernel.
    The same sigma is considered in both spatial axes
    (a_smooth), while a different one can be set for the
    spectral axis (v_smooth).
    Note that the 'stat' cube is convolved as convol^2[statcube].
    The smoothChiCube is than created as:
                   convol[datacube]
    sChiCube = ------------------------
               sqrt(convol**2[statcube])
    The chi_cube is simply:
                       datacube
    chi_cube = -------------------------
                    sqrt(statcube)
    Given that scipy has hard time to deal with NaNs, NaNs
    values in the dataCube are converted to zeroes, while the
    NaNs in the statCube to nanmax(statcube).

    Parameters
    ----------
    datacontainer : IFUcube, np.array
        data initialized in cubeClass.py, or data in 3D array resulting from subtractBg
    min_lambda : int
        minimum wavelength to collapse data
    max_lambda : int
        maximum wavelength to collapse data
    statcube : np.array, optional
        variance in 3D array, resulting array from subtractBg
    a_smooth : float
        smooth length in pixel in the spatial direction
    v_smooth : float
        smooth length in pixel in the velocity direction
    truncate : float
        number of sigma after which the smoothing kernel
        is truncated

    Returns
    -------
    chi_cube, sChiCube : np.array
        X-cube and smoothed X-cube
    """

    print("smoothChiCube: Shrinking Cube with given parameters")
    if statcube is None:
        datacube, statcube = datacontainer.get_data_stat()
    else:
        datacube = datacontainer
    datacube = datacube[min_lambda:max_lambda, :, :]
    statcube = statcube[min_lambda:max_lambda, :, :]

    print("smoothChiCube: Smoothing Cube with 3D Gaussian Kernel")

    data_sigma = (v_smooth, a_smooth, a_smooth)
    stat_sigma = (v_smooth / np.sqrt(2.), a_smooth / np.sqrt(2.), a_smooth / np.sqrt(2.))

    # Removing nans
    datacopy = np.nan_to_num(datacube, copy='True')
    statcopy = np.copy(statcube)
    statcopy[np.isnan(statcube)] = np.nanmax(statcube)

    # smooth cubes
    data_smooth = ndimage.filters.gaussian_filter(datacopy, data_sigma,
                                                  truncate=truncate)
    stat_smooth = ndimage.filters.gaussian_filter(statcopy, stat_sigma,
                                                  truncate=truncate)

    # Cleaning up memory
    del data_sigma
    del stat_sigma
    gc.collect()

    return datacopy / np.sqrt(statcopy), data_smooth / np.sqrt(stat_smooth)


def maskHalo(chi_cube,
             x_pos,
             y_pos,
             z_pos,
             rad_bad_pix,
             rad_max,
             r_connect=2,
             threshold=2.,
             threshold_type='relative',
             bad_pixel_mask=None,
             n_sigma_extreme=5.,
             output='Object',
             debug=False,
             showDebug=False):
    """Given a PSF subtracted data cube (either smoothed or not) this
    macro, after masking some regions, performs a friends of friends
    search for connected pixels above a certain threshold in S/N.
    The first step is to identify the most significant voxel in
    proximity of the quasar position (given as x_pos, y_pos, z_pos).
    The code assumes that the position of the extended halo is known
    and so started to create the mask of connected pixels from this
    point and from the most significant voxel within a spherical
    radius of 3.*rad_bad_pix from (x_pos, y_pos, z_pos).
    From there the macro searches for neighbor pixels that are above
    the threshold and creates a mask for the extended emission.

    Parameters
    ----------
    chi_cube : np.array
        3D X cube output of smoothChiCube. This is constructed
        as data/noise (or as smooth(data)/smooth(noise)).
    x_pos, y_pos, z_pos : float
        position from where start to search for the presence
        of a halo. This is the position of the quasar in x and y
        and the expected position of the extended halo at the
        quasar redshift in z.
    rad_bad_pix : int
        radius of the circular region centred in x_pos, y_pos that
        will be masked. This is typically due to the absence of
        information at the quasar location caused by the normalization
        of the empirical PSF model to the central region.
    rad_max : int
        the circular region centred in x_pos and y_pos with radius
        rad_max will be masked and not considered in the search of
        extended emission. This helps to prevent boundary problems
        and to speed up the algorithm.
    r_connect : int
        default is rConnect=2. Connecting distance used in the FoF
        algorithm.
    threshold : float
        S/N threshold to consider a pixel as part of the extended
        emission. Default is 2.
    threshold_type : str
        'relative': A pixel will be considered as a part of a halo
        if it is above 'threshold' times the sigma of the distribution
        of S/N of the pixels.
        'absolute' : A pixel will be considered as a part of a halo
        if it is above the value of 'threshold'.
    bad_pixel_mask : np.array, or mask of boolean
        2D mask to remove spatial pixels from the estimate of the halo
        location. If 1 (or True) the spatial pixel will be removed.
    n_sigma_extreme : float, optional
        parameter for statFullCube and statFullCubeZ:
        if not None, voxels with values larger than
        sigmaExtreme times the standard deviation of
        the cube will be masked (default is 5.)
    output : string
        root file name for outputs

    Returns
    -------
    mask_halo : np.array
        mask where the detected extended emission is set to 1 and
        the background is set to 0. It has the same shape of the
        input chi_cube.
    """

    # Creating a mask
    print("maskHalo: removing inner region around (x,y)=({:.2f},{:.2f})".format(float(x_pos), float(y_pos)))
    print("          with radius {} pixels".format(rad_bad_pix))
    bad_mask = manip.location(chi_cube[0, :, :], x_position=x_pos, y_position=y_pos,
                              semi_maj=rad_bad_pix, semi_min=rad_bad_pix)

    print("maskHalo: removing outer region around (x,y)=({:.2f},{:.2f})".format(float(x_pos), float(y_pos)))
    print("          with radius {} pixels".format(rad_max))
    outer_bad_mask = manip.location(chi_cube[0, :, :], x_position=x_pos, y_position=y_pos,
                                    semi_maj=rad_max, semi_min=rad_max)
    bad_mask[(outer_bad_mask == 0)] = 1
    del outer_bad_mask

    if bad_pixel_mask is not None:
        print("maskHalo: removing {} bad voxels".format(np.sum(bad_pixel_mask)))
        bad_mask[(bad_pixel_mask > 0)] = 1

    # Filling up mask
    chicopy = np.copy(chi_cube)
    chi_cube_mask = np.zeros_like(chi_cube)
    channel_array = manip.channel_array(chicopy, 'z')
    z_max, y_max, x_max = np.shape(chi_cube)
    for channel in channel_array:
        chicopy[channel, :, :][(bad_mask == 1)] = np.nan
        chi_cube_mask[channel, :, :][(bad_mask == 1)] = 1

    print("maskHalo: defining threshold level")

    # In absence of systematics, the distribution of the
    # X values in a single slice should be a gaussian
    # centered in 0 with sigma=1. This is not true in
    # case the smoothed cube is used and/or if correlation
    # between voxels are present in the cubes.
    # To calculate the threshold, the macro does an
    # approximation, assuming that the distribution is
    # more or less Gaussian. This is NOT true, and will
    # be fixed in the future.

    chi_cube_ave, chi_cube_med, chi_cube_sig = manip.statFullCube(chicopy, n_sigma_extreme=n_sigma_extreme)
    chi_cube_ave_z, chi_cube_med_z, chi_cube_sig_z = manip.statFullCubeZ(chicopy, n_sigma_extreme=n_sigma_extreme)
    print("maskHalo: the median value of the voxels is: {:+0.4f}".format(chi_cube_med))
    print("          and the sigma is: {:+0.4f}".format(chi_cube_sig))
    if threshold_type == 'relative':
        print("maskHalo: the average relative threshold value set to {:0.2f}*{:0.4f}={:0.4f}".format(threshold,
                                                                                                      chi_cube_sig,
                                                                                                      threshold *
                                                                                                      chi_cube_sig))
        threshold_halo = threshold * chi_cube_sig_z
    elif threshold_type == 'absolute':
        print("maskHalo: absolute threshold value set to {:0.2f}".format(threshold))
        threshold_halo = threshold * np.ones_like(chi_cube_sig_z)
    else:
        print("maskHalo: WARNING!")
        print("          no threshold_type set, assumed relative")
        print("maskHalo: the average relative threshold value set to {:0.2f}*{:0.4f}={:0.4f}".format(threshold,
                                                                                                      chi_cube_sig,
                                                                                                      threshold *
                                                                                                      chi_cube_sig))
        threshold_halo = threshold * chi_cube_sig_z

    if debug:
        print("maskHalo: Saving debug image on {}_voxelDistribution.pdf".format(output))
        print("          in principle the distribution should be gaussian")
        print("          showing only channel {}".format(np.int(z_max / 2.)))

        manip.nicePlot()

        # Populating the histogram
        chi_cube_hist, chi_cube_edges = np.histogram(
            chicopy[np.int(z_max / 2.), :, :][np.isfinite(chicopy[1, :, :])].flatten(),
            bins="fd", density=True)
        # fitting of the histogram
        chi_cube_edges_bin = np.nanmedian(chi_cube_edges - np.roll(chi_cube_edges, 1))
        gauss_best, gauss_covar = curve_fit(manip.gaussian,
                                            chi_cube_edges[:-1],
                                            chi_cube_hist,
                                            p0=[np.nansum(chi_cube_hist * chi_cube_edges_bin) / (
                                                    chi_cube_sig * np.sqrt((2. * np.pi))),
                                                chi_cube_med, chi_cube_sig])

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_hist = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        ax_image.imshow(chicopy[np.int(z_max / 2.), :, :],
                        cmap="Greys", origin="lower",
                        vmin=chi_cube_med - 3. * chi_cube_sig,
                        vmax=chi_cube_med + 3. * chi_cube_sig)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_title(r"Channel {} X-image".format(int(z_max / 2.)))
        ax_image.set_xlim(left=x_pos - rad_max, right=x_pos + rad_max)
        ax_image.set_ylim(bottom=y_pos - rad_max, top=y_pos + rad_max)

        # Plotting pixel distribution
        ax_hist.step(chi_cube_edges[:-1], chi_cube_hist, color="gray",
                     zorder=3)
        ax_hist.plot(chi_cube_edges[:-1], manip.gaussian(chi_cube_edges[:-1], *gauss_best),
                     color='black', zorder=2)
        ax_hist.axvline(chi_cube_med, color="black",
                        zorder=1, linestyle=':')
        ax_hist.set_xlim(left=chi_cube_med - 3. * chi_cube_sig,
                         right=chi_cube_med + 3. * chi_cube_sig)
        ax_hist.set_ylim(bottom=-0.01 * np.nanmax(manip.gaussian(chi_cube_edges[:-1], *gauss_best)),
                         top=1.2 * np.nanmax(manip.gaussian(chi_cube_edges[:-1], *gauss_best)))
        ax_hist.text(0.52, 0.9, "Median", transform=ax_hist.transAxes)
        ax_hist.set_ylabel(r"Pixel Distribution", size=30)
        ax_hist.set_xlabel(r"X", size=30)
        ax_hist.set_title(r"X values distribution")

        plt.tight_layout()
        plt.savefig(output + "_voxelDistribution.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del chi_cube_hist
        del chi_cube_edges
        del gauss_best
        del gauss_covar

    # Searching for the Max value from which to start to look for connections
    print("maskHalo: searching for extended emission")
    print("          starting from from (x,y,z)=({:0.2f},{:0.3f},{:0.0f})".format(float(x_pos), float(y_pos),
                                                                                  float(z_pos)))
    # Check small cube
    small_chicopy = chicopy[int(z_pos - 3. * rad_bad_pix):int(z_pos + 3. * rad_bad_pix),
                    int(y_pos - 3. * rad_bad_pix):int(y_pos + 3. * rad_bad_pix),
                    int(x_pos - 3. * rad_bad_pix):int(x_pos + 3. * rad_bad_pix)]

    if np.nansum(np.isfinite(small_chicopy)) > 0:
        max_s_chicopy = np.nanmax(small_chicopy)
        # Selecting the closest value
        z_max_chi, y_max_chi, x_max_chi = np.where(chicopy == max_s_chicopy)
        dist_pix = manip.distFromPixel(z_pos, y_pos, x_pos,
                                       z_max_chi, y_max_chi, x_max_chi)
        z_max_chi = int(z_max_chi[np.where(dist_pix == np.min(dist_pix))])
        y_max_chi = int(y_max_chi[np.where(dist_pix == np.min(dist_pix))])
        x_max_chi = int(x_max_chi[np.where(dist_pix == np.min(dist_pix))])

        print("maskHalo: the maximum S/N detected is {:0.3f} ".format(max_s_chicopy))
        print("          at the location (x,y,z)=({},{},{})".format(x_max_chi, y_max_chi, z_max_chi))
    else:
        max_s_chicopy = threshold
        z_max_chi, y_max_chi, x_max_chi = z_pos, y_pos, x_pos
    del small_chicopy

    print("maskHalo: starting to fill the halo mask")
    # this mask is equal to 1 if the pixel is considered part of the halo
    # equal to 0 if it is not
    mask_halo = np.zeros_like(chi_cube)
    if max_s_chicopy > threshold_halo[z_max_chi]:
        mask_halo[z_max_chi, y_max_chi, x_max_chi] = 1
    # The code also generate a 'seed' at the expected position of the
    # extended halo. This is helpful to avoid to be too dependent on the
    # location of the brightest pixel. It will be removed before output.

    z_min_seed, z_max_seed = int(z_pos - 5. * r_connect), int(z_pos + 5. * r_connect)
    y_min_seed, y_max_seed = int(y_pos - rad_bad_pix), int(y_pos + rad_bad_pix)
    x_min_seed, x_max_seed = int(x_pos - rad_bad_pix), int(x_pos + rad_bad_pix)

    mask_halo[z_min_seed:z_max_seed, y_min_seed:y_max_seed, x_min_seed:x_max_seed] = 1
    chi_cube_mask[z_min_seed:z_max_seed, y_min_seed:y_max_seed, x_min_seed:x_max_seed] = 0

    # here the code start to propagate the mask in the neighbor pixels
    connected_pix = 0
    connected_pix_new = int(np.nansum(mask_halo))

    while connected_pix < connected_pix_new:
        connected_pix = int(np.nansum(mask_halo))
        # create a new mask around the identified voxels
        temp_halo_mask = np.copy(mask_halo)
        z_mask, y_mask, x_mask = np.where(temp_halo_mask == 1)
        for zMaskTemp, yMaskTemp, xMaskTemp in zip(z_mask, y_mask, x_mask):
            z_mask_min, z_mask_max = int(zMaskTemp - r_connect), int(zMaskTemp + r_connect)
            y_mask_min, y_mask_max = int(yMaskTemp - r_connect), int(yMaskTemp + r_connect)
            x_mask_min, x_mask_max = int(xMaskTemp - r_connect), int(xMaskTemp + r_connect)
            temp_halo_mask[z_mask_min:z_mask_max, y_mask_min:y_mask_max, x_mask_min:x_mask_max] = 1

        # check that voxels in maskHaloTemp are above the threshold
        new_chicopy = np.copy(chi_cube)
        new_chicopy[~np.isfinite(new_chicopy)] = 0.
        new_chicopy[(chi_cube_mask == 1)] = 0.
        new_chicopy *= temp_halo_mask.astype(float)

        for channel in np.arange(0, z_max):
            mask_halo[channel, :, :][new_chicopy[channel, :, :] > threshold_halo[channel]] = 1
        connected_pix_new = int(np.nansum(mask_halo))
    # Removing seed
    mask_halo[z_min_seed:z_max_seed, y_min_seed:y_max_seed, x_min_seed:x_max_seed] = 0
    # Removing masked data

    if debug:
        print("maskHalo: Creating debug image")
        print("          Plotting Channel {} where the most significant voxel is.".format(z_max_chi))
        print("          The location of this voxel is marked with a red circle")
        print("          The position of the quasars is in blue")

        manip.nicePlot()

        mask_halo_2D, mask_halo_min_z, mask_halo_max_z = spectral_mask_halo(mask_halo)

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_mask = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        x_plot_min, x_plot_max = int(x_pos - rad_max), int(x_pos + rad_max)
        y_plot_min, y_plot_max = int(y_pos - rad_max), int(y_pos + rad_max)
        std_plot = np.nanstd(chi_cube[z_max_chi,
                             y_plot_min:x_plot_max,
                             x_plot_min:y_plot_max])
        # Plotting field image
        ax_image.imshow(chi_cube[z_max_chi, :, :],
                        cmap="Greys", origin="lower",
                        vmin=-1. * std_plot,
                        vmax=+3. * std_plot)
        ax_mask.contour(mask_halo[z_max_chi, :, :], colors='maroon',
                        alpha=0.9, origin="lower", linewidths=0.5)
        ax_mask.contour(mask_halo_2D, colors='orangered',
                        alpha=0.5, origin="lower", linewidths=0.5)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_xlim(left=x_plot_min, right=x_plot_max)
        ax_image.set_ylim(bottom=y_plot_min, top=y_plot_max)
        ax_image.set_title(r"Channel {} Map".format(z_max_chi))

        max_artist = Ellipse(xy=(x_max_chi, y_max_chi),
                             width=r_connect,
                             height=r_connect,
                             angle=0.)
        max_artist.set_facecolor("none")
        max_artist.set_edgecolor("red")
        max_artist.set_alpha(0.5)
        ax_image.add_artist(max_artist)

        qso_artist = Ellipse(xy=(x_pos, y_pos),
                             width=rad_bad_pix,
                             height=rad_bad_pix,
                             angle=0.)
        qso_artist.set_facecolor("none")
        qso_artist.set_edgecolor("blue")
        qso_artist.set_alpha(0.5)
        ax_image.add_artist(qso_artist)

        # Plotting mask image
        ax_mask.imshow(bad_mask,
                       cmap="Greys", origin="lower",
                       vmin=0.,
                       vmax=.5)
        ax_mask.contour(mask_halo[z_max_chi, :, :], colors='maroon',
                        alpha=0.9, origin="lower", linewidths=0.5)
        ax_mask.contour(mask_halo_2D, colors='orangered',
                        alpha=0.5, origin="lower", linewidths=0.5)
        ax_mask.set_xlabel(r"X [Pixels]", size=30)
        ax_mask.set_ylabel(r"Y [Pixels]", size=30)
        ax_mask.set_xlim(left=x_pos - rad_max, right=x_pos + rad_max)
        ax_mask.set_ylim(bottom=y_pos - rad_max, top=y_pos + rad_max)
        ax_mask.set_title(r"Excluded Pixels Mask")

        print("maskHalo: debug image saved in {}_maskHaloStartingVoxel.pdf".format(output))
        plt.tight_layout()
        plt.savefig(output + "_maskHaloStartingVoxel.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del mask_halo_2D
        del mask_halo_min_z
        del mask_halo_max_z

    # Clearing up memory
    del bad_mask
    del new_chicopy
    del chicopy
    del chi_cube_mask

    return mask_halo


def spectral_mask_halo(mask_halo):
    """Given the halo mask, this macro returns
    a 2D mask in x and y and the min and max
    channel where the halo is detected in the z-axis.

    Parameters
    ----------
    mask_halo : np.array
        3D mask of the halo location.

    Returns
    -------
    mask_halo_2D : np.array
        2D mask of the halo location (collapsed along the
        z-axis).
    maskHaloMinZ, maskHaloMaxZ : int
        min and max channel where the halo is detected along
        the z-axis
    """

    print("spectral_mask_halo: collapsing halo mask")
    if np.nansum(mask_halo) < 2:
        print("spectral_mask_halo: not enough voxels in the mask")
        z_max, y_max, x_max = np.shape(mask_halo)
        return np.zeros_like(mask_halo[0, :, :], int), 0, z_max

    mask_halo_2D = np.nansum(mask_halo, axis=0)
    mask_halo_2D[(mask_halo_2D > 0)] = 1

    # Collapsing along 1,2 to obtain the z-map
    mask_halo_z = np.nansum(mask_halo, axis=(1, 2))
    z_max, y_max, x_max = np.shape(mask_halo)
    channel = np.arange(0, z_max, 1, int)
    mask_halo_min_z, mask_halo_max_z = np.nanmin(channel[mask_halo_z > 0]), np.nanmax(channel[mask_halo_z > 0])

    # Cleaning up memory
    del mask_halo_z
    del channel

    return mask_halo_2D, mask_halo_min_z, mask_halo_max_z


def clean_mask_halo(mask_halo, delta_z_min=2, min_vox=100,
                    min_channel=None, max_channel=None,
                    debug=False, showDebug=False):
    """
    Given the halo mask, the macro performs some quality
    check:
    * If a spatial pixel (x,y) has less than delta_z_min consecutive voxels identified along the z-axis, this will
    be removed from the mask
    * If the total number of voxels is less than min_vox the halo is considered as not detected and the mask is cleaned.

    Parameters
    ----------
    mask_halo : np.array
        3D mask of the halo location.
    delta_z_min : int, optional
        min size in the spectral axis to consider the voxel
        as part of the halo
    min_vox : int, optional
        voxel detection threshold value. If the number of voxels is less than this parameter,
        the halo is not to be detected (default 100)
    min_channel, max_channel : np.array
        only voxels between channelMin and max_channel in the
        spectral direction will be considered in the creation
        of the cleanMask

    Returns
    -------
    maskHaloClean : np.array
        cleaned 3D mask of the halo location.
    """

    print("clean_mask_halo: cleaning halo mask")

    if np.nansum(mask_halo) < np.int(min_vox / 2.):
        print("clean_mask_halo: not enough voxels in the mask")
        return np.zeros_like(mask_halo, int)

    clean_mask_halo = np.copy(mask_halo)
    # Collapsing along 1,2 to obtain the z-map
    maskHaloCleanXY = np.nansum(clean_mask_halo, axis=0)
    z_max, y_max, x_max = np.shape(clean_mask_halo)
    for channel in np.arange(0, z_max, 1, int):
        clean_mask_halo[channel, :, :][(maskHaloCleanXY <= delta_z_min)] = 0
    if np.nansum(clean_mask_halo) <= min_vox:
        clean_mask_halo[:, :, :] = 0
    if min_channel is not None:
        clean_mask_halo[0:min_channel, :, :] = 0
    if max_channel is not None:
        clean_mask_halo[max_channel:-1, :, :] = 0

    print("clean_mask_halo: Removed {} voxels from the mask".format(np.sum(mask_halo) - np.sum(clean_mask_halo)))

    if debug:
        z_max, y_max, x_max = np.shape(mask_halo)
        channel_y = np.arange(0, y_max, 1, int)
        channel_x = np.arange(0, x_max, 1, int)
        mask_halo_y = np.nansum(mask_halo, axis=(0, 2))
        mask_halo_x = np.nansum(mask_halo, axis=(0, 1))
        min_y_mask, max_y_mask = np.nanmin(channel_y[mask_halo_y > 0]), np.nanmax(channel_y[mask_halo_y > 0])
        min_x_mask, max_x_mask = np.nanmin(channel_x[mask_halo_x > 0]), np.nanmax(channel_x[mask_halo_x > 0])
        mask_halo_2D, _, _ = spectral_mask_halo(mask_halo)
        clean_mask_2D, _, _ = spectral_mask_halo(clean_mask_halo)
        plt.figure(1, figsize=(6, 6))
        plt.contour(mask_halo_2D, colors='blue',
                    alpha=0.9, origin="lower")
        plt.contour(clean_mask_2D, colors='red',
                    alpha=0.9, origin="lower")
        plt.xlim(min_x_mask - 2, max_x_mask + 2)
        plt.ylim(min_y_mask - 2, max_y_mask + 2)
        plt.xlabel(r"X [Pixels]", size=30)
        plt.ylabel(r"Y [Pixels]", size=30)
        if showDebug:
            plt.show()
        plt.close()
        del mask_halo_2D
        del clean_mask_2D
        del mask_halo_y
        del mask_halo_x
    # Cleaning up memory
    del maskHaloCleanXY

    return clean_mask_halo


def makeMoments(datacontainer,
                mask_halo,
                central_wave=None,
                s_smooth=None,
                truncate=5.,
                debug=False,
                showDebug=False):
    """ Given a PSF-Subtracted datacube, this macro extracts the moment 0, 1, 2
    maps of the halo identified by mask_halo.
    Where:

    * mom0: is the integrated value

    .. math::
        mom0 = sum[Flux*dlambda]
    where sum is the sum along the channels

    * mom1: is the velocity field
     .. math::
         mom1 = sum[DV*Flux] / sum[Flux]

    where DV is the velocity difference of a channel from the centralWave
     .. math::
         DV = (wavelength - centralWave) / centralWave * speed of light

    * mom2: is the velocity dispersion
    .. math::
         mom2 = sqrt[ sum[ Flux*(DV-mom1)**2. ] / sum[Flux] ]

    Parameters
    ----------
    datacontainer : IFUcube object
        data initialized in the cubeClass.py file
    mask_halo : np.array
        mask where the detected extended emission is set to 1 and the background is set to 0. It has the same shape
        of the input datacube.
    central_wave : float
        wavelength in Ang. from which to calculate the
        velocity shifts. If None, the macro will calculate
        it from the spectrum of the halo.
    s_smooth : float
        smooth length in pixel in the spatial direction
    truncate : float
        number of sigma after which the smoothing kernel
        is truncated

    Returns
    -------
    mom0, mom1, mom2 : np.array
        moment maps in 2D arrays. Units for mom0 are
        of fluxes, while for mom1 and mom2 are of
        velocity
    centralWave : float
        wavelength in Ang. from which to the velocity shifts
        are calculated. It is equal to the input wavelength
        if not set to None.
    """
    print('makeMonents: reading in IFUcube')
    headers = datacontainer.get_primary()
    datacopy, statcopy = datacontainer.get_data_stat()
    print("makeMoments: estimating halo moments")

    if np.nansum(mask_halo) < 3:
        print("makeMoments: not enough voxels to calculate moment maps")
        central_wave = 0.
        null_image = np.zeros_like(datacopy[0, :, :], float)
        return null_image, null_image, null_image, central_wave

    tmp_datacopy = np.copy(datacopy)

    gkernel_radius = 0
    while np.nansum(~np.isfinite(tmp_datacopy)) > 0:
        gkernel_radius = gkernel_radius + 2
        print("makeMoments: masking {} NaNs with a {} spatial pixel Gaussian Kernel".format(
            np.nansum(~np.isfinite(tmp_datacopy)), gkernel_radius))
        print("             the total number of voxels is {}".format(np.size(tmp_datacopy)))
        gkernel = astropy.convolution.Gaussian2DKernel(gkernel_radius)
        z_max, y_max, x_max = np.shape(datacopy)
        for channel in np.arange(0, z_max):
            data_channel = np.copy(tmp_datacopy[channel, :, :])
            # correct for nan with blurred images
            blur_channel = astropy.convolution.convolve(np.copy(datacopy[channel, :, :]), gkernel)
            data_channel[np.logical_not(np.isfinite(data_channel))] = blur_channel[
                np.logical_not(np.isfinite(data_channel))]
            tmp_datacopy[channel, :, :] = data_channel[:, :]
        del blur_channel, data_channel

    if s_smooth is not None:
        print("makeMoments: smoothing cube")
        v_smooth = 0
        data_sigma = (v_smooth, s_smooth, s_smooth)
        # Removing nans
        tmp_datacopy = np.nan_to_num(datacopy, copy=True)
        tmp_datacopy = ndimage.filters.gaussian_filter(tmp_datacopy, data_sigma,
                                                       truncate=truncate)
    # Extract Spectrum
    mask_halo_2D, min_z_mask, max_z_mask = spectral_mask_halo(mask_halo)
    flux_halo, err_flux_halo = manip.quickSpectrumNoBgMask(datacontainer,
                                                           mask_halo_2D)
    flux_halo_smooth, err_flux_halo_smooth = manip.quickSpectrumNoBgMask(datacontainer,
                                                                         mask_halo_2D)

    # removing voxels outside the halo
    tmp_datacopy[(mask_halo < 1)] = np.float(0.)

    # Defining wavelength range
    z_max, y_max, x_max = np.shape(datacopy)
    channels = np.arange(0, z_max)
    wave = manip.convert_to_wave(headers, channels)

    # find central wavelength of the halo:
    # Extract "optimally extracted" spectrum
    optimal_flux_halo = np.nansum(tmp_datacopy, axis=(1, 2))

    if central_wave is None:
        central_wave = np.nansum(wave * optimal_flux_halo) / np.nansum(optimal_flux_halo)
    print("makeMoments: the central wavelength considered is {:+.2f}".format(central_wave))

    # Moment Zero
    mom0 = manip.collapse_cube(tmp_datacopy)

    # Moment One
    vel_shift = (wave - central_wave) / central_wave * cSpeed  # in km/s
    vel_shift_cube = np.zeros_like(tmp_datacopy, float)
    for channel in np.arange(0, z_max):
        vel_shift_cube[channel, :, :] = vel_shift[channel]
    mom1 = np.nansum(vel_shift_cube * tmp_datacopy, axis=0) / np.nansum(tmp_datacopy, axis=0)

    # Moment Two
    sigma_cube = np.ones_like(tmp_datacopy, float)
    for channel in np.arange(0, z_max):
        sigma_cube[channel, :, :] = np.power(vel_shift[channel] - mom1[:, :], 2.)
    mom2 = np.sqrt(np.nansum(sigma_cube * tmp_datacopy, axis=0) / np.nansum(tmp_datacopy, axis=0))

    if debug:

        print("makeMoments: showing debug image:")
        print("             spectrum and optimally extracted spectrum")

        manip.nicePlot()

        # Setting limits for halo image
        channel_y = np.arange(0, y_max, 1, int)
        channel_x = np.arange(0, x_max, 1, int)
        mask_halo_y = np.nansum(mask_halo, axis=(0, 2))
        mask_halo_x = np.nansum(mask_halo, axis=(0, 1))
        min_y_mask, max_y_mask = np.nanmin(channel_y[mask_halo_y > 0]), np.nanmax(channel_y[mask_halo_y > 0])
        min_x_mask, max_x_mask = np.nanmin(channel_x[mask_halo_x > 0]), np.nanmax(channel_x[mask_halo_x > 0])

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        ax_imag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

        ax_imag.imshow(mom0,
                       cmap="Greys", origin="lower")
        ax_imag.set_xlim(left=min_x_mask, right=max_x_mask)
        ax_imag.set_ylim(bottom=min_y_mask, top=max_y_mask)
        ax_imag.set_xlabel(r"X [Pixels]", size=30)
        ax_imag.set_ylabel(r"Y [Pixels]", size=30)

        ax_spec.plot(wave, flux_halo, color='black', zorder=3, label='Total')
        ax_spec.plot(wave, flux_halo_smooth, color='gray', zorder=3, label='Smooth')
        ax_spec.plot(wave, optimal_flux_halo, color='red', zorder=3, label='Opt. Ex.')
        ax_spec.plot(wave, err_flux_halo, color='gray', alpha=0.5, zorder=2, label='Error')
        ax_spec.legend(loc='upper left', ncol=2, fontsize=15)
        ax_spec.set_xlabel(r"Wavelength", size=30)
        ax_spec.set_ylabel(r"Flux", size=30)
        ax_spec.axvline(central_wave, color="red",
                        zorder=1, linestyle='-')
        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()

        print("makeMoments: showing debug image:")
        print("             0, 1, and 2 moments")

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        ax_mom0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax_mom1 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
        ax_mom2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

        img_mom0 = ax_mom0.imshow(mom0,
                                  cmap="Greys", origin="lower")
        ax_mom0.set_xlim(left=min_x_mask, right=max_x_mask)
        ax_mom0.set_ylim(bottom=min_y_mask, top=max_y_mask)
        ax_mom0.set_xlabel(r"X [Pixels]", size=30)
        ax_mom0.set_ylabel(r"Y [Pixels]", size=30)
        cbMom0 = plt.colorbar(img_mom0, ax=ax_mom0, shrink=0.5)

        img_mom1 = ax_mom1.imshow(mom1,
                                  cmap="Greys", origin="lower",
                                  vmin=-0.5 * np.nanmax(mom1),
                                  vmax=+0.5 * np.nanmax(mom1))
        ax_mom1.set_xlim(left=min_x_mask, right=max_x_mask)
        ax_mom1.set_ylim(bottom=min_y_mask, top=max_y_mask)
        ax_mom1.set_xlabel(r"X [Pixels]", size=30)
        ax_mom1.set_ylabel(r"Y [Pixels]", size=30)
        cbMom1 = plt.colorbar(img_mom1, ax=ax_mom1, shrink=0.5)

        img_mom2 = ax_mom2.imshow(mom2,
                                  cmap="Greys", origin="lower",
                                  vmin=0.,
                                  vmax=+0.9 * np.nanmax(mom2))
        ax_mom2.set_xlim(left=min_x_mask, right=max_x_mask)
        ax_mom2.set_ylim(bottom=min_y_mask, top=max_y_mask)
        ax_mom2.set_xlabel(r"X [Pixels]", size=30)
        ax_mom2.set_ylabel(r"Y [Pixels]", size=30)
        cbMom2 = plt.colorbar(img_mom2, ax=ax_mom2, shrink=0.5)

        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()

    # Cleaning up memory
    del mask_halo_2D, min_z_mask, max_z_mask
    del tmp_datacopy
    del flux_halo, err_flux_halo
    del flux_halo_smooth, err_flux_halo_smooth
    del wave
    del vel_shift_cube, sigma_cube
    gc.collect()

    return mom0, mom1, mom2, central_wave


def maxExtent(mask_halo_2D, x_pos=None, y_pos=None):
    """Given a 2D mask halo, the macro calculates
    the maximum extend of the halo in the spatial
    direction and the maximum distance from x_pos,
    y_pos (if given).

    Parameters
    ----------
    mask_halo_2D : np.array
        2D mask of the halo location

    Returns
    -------
    maxExtent : float
        maximum extent in pixels
    fromPixExtent : float
        maximum extent from (x_pos, y_pos) in pixels
    """

    print("maxExtent: estimating maximal extent")

    if np.nansum(mask_halo_2D) < 3:
        print("maxExtent: not enough spatial pixels")
        return 0., 0.

    max_extent = 0.

    y_pivot, x_pivot = np.where(mask_halo_2D > 0)
    z_pivot = np.zeros_like(y_pivot, int)
    for z_index, y_index, x_index in zip(z_pivot, y_pivot, x_pivot):
        temp_max_ext = np.nanmax(manip.distFromPixel(z_index, y_index, x_index, z_pivot, y_pivot, x_pivot))
        if temp_max_ext > max_extent:
            max_extent = temp_max_ext

    if x_pos is None:
        return float(max_extent), 0.

    from_pix_extent = np.nanmax(manip.distFromPixel(float(0.), y_pos, x_pos, z_pivot, y_pivot, x_pivot))

    return float(max_extent), float(from_pix_extent)
