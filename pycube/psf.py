import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import gridspec

import numpy as np
from scipy.optimize import curve_fit

from pycube.core import manip
from pycube.core import background

from astropy.stats import sigma_clip

import sep
import gc


def find_sources(datacontainer, statcube=None,
                 min_lambda=None, max_lambda=None,
                 mask_z=None,
                 var_factor=5.,
                 threshold=4.,
                 sig_detect=2.,
                 min_area=3.,
                 gain=1.1,  # get it from the header
                 deblend_val=0.005):
    """Automated scanning of given data and identifies good sources.
    If data is in 3D format, function will collapse given wavelength parameters

    Parameters
    ----------
    datacontainer : np.array, IFUcube object
        3D data cube of or data file initialized in cubeClass.py
    statcube : np.array, optional
        3D variance of data, assumes IFUcube for datacube (default is None)
    min_lambda : float
        minimum wavelength value to collapse 3D image (default is None)
    max_lambda : float
        maximum wavelength value to collapse 3D image (default is None)
    mask_z : np.array
        range of z-axis values that the user can mask in the datacube during collapsing (default is None)
    var_factor : int, float, optional
        affects generated variance, if variance is auto-generated from image data (default is 5.)
    threshold : int, float, optional
        threshold value for sigma detection with seXtractor background subtraction (default is 4.)
    sig_detect : int, float, optional
        minimum signal detected by function (default is 2.)
    min_area : int, float, optional
        minimum area determined to be a source (default is 3.)
    gain : float, optional
        can be pulled from Fits file (default is 1.1)
    deblend_val : float, optional
            value for sep extractor, minimum contrast ratio for object blending (default is 0.005)

    Returns
    -------
    x_pos : np.array
        x positions of objects
    y_pos : np.array
        y positions of objects
    maj_axis : np.array
        semi-major axis value of objects
    min_axis : np.array
        semi-minor axis of objects
    angle : np.array
        angle for elliptical masking
    all_objects : np.array
        directory for all objects found and their values
    """

    # correct collapsing if object passed or two 3D arrays
    if statcube is None:
        data_background, var_background = manip.collapse_container(datacontainer, min_lambda=min_lambda,
                                                                   max_lambda=max_lambda, mask_z=mask_z,
                                                                   var_thresh=var_factor)
    else:
        datacopy = datacontainer
        data_background = manip.collapse_cube(datacopy, min_lambda=min_lambda, max_lambda=max_lambda, mask_z=mask_z)
        statcopy = statcube
        var_background = manip.collapse_cube(statcopy, min_lambda=min_lambda, max_lambda=max_lambda, mask_z=mask_z)

    image_background = background.sextractor_background(data_background, var_background, threshold)
    void_background = data_background - image_background

    # print("find_sources: Searching sources {}-sigma above noise".format(sig_detect))
    all_objects = sep.extract(void_background, sig_detect,
                              err=var_background,
                              minarea=min_area,
                              filter_type='matched',
                              # gain=gain,
                              clean=True,
                              deblend_cont=deblend_val,
                              filter_kernel=None)
    del data_background
    del var_background
    # Sorting sources by flux at the peak
    index_by_flux = np.argsort(all_objects['peak'])[::-1]
    all_objects = all_objects[index_by_flux]
    good_sources = all_objects['flag'] < 1
    x_pos = np.array(all_objects['x'][good_sources])
    y_pos = np.array(all_objects['y'][good_sources])
    maj_axis = np.array(all_objects['a'][good_sources])
    min_axis = np.array(all_objects['b'][good_sources])
    angle = np.array(all_objects['theta'][good_sources])
    # print("find_sources: {} good sources detected".format(np.size(x_pos)))
    # preserve memory
    del image_background
    del void_background
    del good_sources
    return x_pos, y_pos, maj_axis, min_axis, angle, all_objects


def background_cube(datacontainer,
                    sig_source_detect=3.,
                    min_source_area=12.,
                    source_mask_size=6.,
                    edges=50):
    """Creates two datacubes from an original IFUcube. One of the background with removed sources
    and smoothed background. Another with the associated mask used for removal.

    Parameters
    ----------
    datacontainer : IFUcube object
        data initialized in cubeClass
    sig_source_detect : float, int, optional
        value of source significance to be detected (default is 3.)
    min_source_area : float, int, optional
        minimum area of sources to be considered for masking (default is 12.)
    source_mask_size : float, int, optional
        mask size (default is 6.)
    edges : int, float
        pixel count to be removed from the edges of the image (default is 50)

    Returns
    -------
    cube_bg : np.array
        data cube of the mask removed background
    mask_bg : np.array
        data cube of the masks implemented
    """

    datacopy = datacontainer.get_data()
    z_max, y_max, x_max = np.shape(datacopy)
    cube_bg = np.full_like(datacopy, np.nan)
    mask_bg = np.copy(cube_bg)
    for index in range(z_max):
        tmpdatacopy = np.copy(datacopy[index, :, :])
        x_pos, y_pos, maj_axis, min_axis, angle, all_objects = find_sources(datacontainer,
                                                                            min_lambda=index,
                                                                            max_lambda=index,
                                                                            sig_detect=sig_source_detect,
                                                                            min_area=min_source_area)
        mask_bg_2D = np.zeros_like(tmpdatacopy)
        sky_mask = manip.location(tmpdatacopy,
                                  x_position=x_pos, y_position=y_pos,
                                  semi_maj=source_mask_size * maj_axis,
                                  semi_min=source_mask_size * min_axis,
                                  theta=angle)

        mask_bg_2D[(sky_mask == 1)] = 1

        edges_mask = np.ones_like(sky_mask, int)
        edges_mask[int(edges):-int(edges), int(edges):-int(edges)] = 0
        mask_bg_2D[(edges_mask == 1)] = 1

        mask_bg_3D = np.broadcast_to((mask_bg_2D == 1), tmpdatacopy.shape)

        tmpdatacopy[(mask_bg_3D == 1)] = np.nan
        bg_datacopy = np.copy(tmpdatacopy)
        bg_datacopy[(mask_bg_2D == 1)] = np.nan

        cube_bg[index] = bg_datacopy
        mask_bg[index] = mask_bg_2D

        # preserve memory
        del tmpdatacopy
        del bg_datacopy
        del mask_bg_2D
        del x_pos, y_pos, maj_axis, min_axis, angle, all_objects

    return cube_bg, mask_bg


def statBg(datacontainer,
           min_lambda=None,
           max_lambda=None,
           mask_z=None,
           mask_xy=None,
           sig_source_detect=5.0,
           min_source_area=16.,
           source_mask_size=6.,
           edges=60,
           output='Object',
           debug=False,
           showDebug=False):
    """This estimates the sky background of a MUSE cube after removing sources.
    Sources are detected in an image created by collapsing the cube between min_lambda
    and max_lambda (considering mask_z as mask for bad channels).
    Average, std, and median will be saved.

    Parameters
    ----------
    datacontainer : IFUcube object
        data initialized in cubeClass.py
    min_lambda : int, optional
        min channel to create the image where to detect sources
    max_lambda : int, optional
        max channel to create the image where to detect sources
    mask_z : np.array, optional
        for collapse function to removed channels masked as 1 (default is None)
    mask_xy : int, bool, optional
        when 1 (or True), this spatial pixel will remove from
        the estimate of the b/g values (default is None)
    sig_source_detect : float, optional
        detection sigma threshold for sources in the
        collapsed cube (default is 5.0)
    min_source_area : float, optional
        min area for source detection in the collapsed
        cube (default is 16)
    source_mask_size : float, optional
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection (default is 6.)
    edges : int, optional
        frame size removed to avoid problems related to the edge
        of the image (default is 60)
    output : string
        root file name for output (default is 'Object')

    Returns
    -------
    averageBg, medianBg, stdBg, varBg, pixelsBg : np.array
        average, median, standard deviation, variance, and number of pixels after masking
        sources, edges, and NaNs of the background.
    maskBg2D : np.array
        2D mask used to determine the background region. This mask has 1 if there is
        a source or is on the edge of the cube. It is 0 if the pixel is considered
        in the background estimate.
    """

    print("statBg: Starting estimate of b/g stats")
    datacopy = datacontainer.get_data()

    data_background = manip.collapse_cube(datacopy, max_lambda=max_lambda, min_lambda=min_lambda, mask_z=mask_z)
    print("statBg: Searching for sources in the collapsed cube")
    x_pos, y_pos, maj_axis, min_axis, angle, all_objects = find_sources(datacontainer, min_lambda=min_lambda,
                                                                        max_lambda=max_lambda,
                                                                        sig_detect=sig_source_detect,
                                                                        min_area=min_source_area)

    print("statBg: Detected {} sources".format(len(x_pos)))
    print("statBg: Masking sources")
    mask_bg_2D = np.zeros_like(data_background)

    sky_mask = manip.location(data_background, x_position=x_pos, y_position=y_pos,
                              semi_maj=source_mask_size * maj_axis,
                              semi_min=source_mask_size * min_axis,
                              theta=angle)

    mask_bg_2D[(sky_mask == 1)] = 1

    print("statBg: Masking Edges")
    # removing edges. This mask is 0 if it is a good pixel, 1 if it is a
    # pixel at the edge
    edges_mask = np.ones_like(sky_mask, dtype=int)
    edges_mask[int(edges):-int(edges), int(edges):-int(edges)] = 0
    mask_bg_2D[(edges_mask == 1)] = 1

    if mask_xy is not None:
        print("statBg: Masking spatial pixels from input mask_xy")
        mask_bg_2D[(mask_xy == 1)] = 1

    print("statBg: Performing b/g statistic")
    mask_bg_3D = np.broadcast_to((mask_bg_2D == 1), datacopy.shape)
    datacopy[(mask_bg_3D == 1)] = np.nan
    average_bg, std_bg, median_bg, var_bg, pixels_bg = np.nanmean(datacopy, axis=(1, 2)), \
                                                       np.nanstd(datacopy, axis=(1, 2)), \
                                                       np.nanmedian(datacopy, axis=(1, 2)), \
                                                       np.nanvar(datacopy, axis=(1, 2)), \
                                                       np.count_nonzero(~np.isnan(datacopy), axis=(1, 2))
    bg_datacopy = np.copy(data_background)
    bg_datacopy[(mask_bg_2D == 1)] = np.nan

    if debug:
        print("statBg: Saving debug image on {}_BgRegion.pdf".format(output))
        bg_flux_copy = np.nanmean(bg_datacopy)
        bg_std_copy = np.nanstd(bg_datacopy)

        plt.figure(1, figsize=(12, 6))

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_clean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        ax_image.imshow(data_background,
                        cmap="Greys", origin="lower",
                        vmin=bg_flux_copy - 3. * bg_std_copy,
                        vmax=bg_flux_copy + 3. * bg_std_copy)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_title(r"Collapsed image")

        # Plotting background image
        ax_clean.imshow(bg_datacopy,
                        cmap="Greys", origin="lower",
                        vmin=bg_flux_copy - 3. * bg_std_copy,
                        vmax=bg_flux_copy + 3. * bg_std_copy)
        ax_clean.set_xlabel(r"X [Pixels]", size=30)
        ax_clean.set_ylabel(r"Y [Pixels]", size=30)
        ax_clean.set_title(r"b/g image")

        plt.tight_layout()
        plt.savefig(output + "_BgRegion.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()

    # preserve memory
    del data_background
    del mask_bg_3D
    del sky_mask
    del edges_mask
    del all_objects
    del datacopy
    gc.collect()

    return average_bg, median_bg, std_bg, var_bg, pixels_bg, mask_bg_2D, bg_datacopy


def subtractBg(datacontainer,
               min_lambda=None,
               max_lambda=None,
               mask_z=None,
               mask_xy=None,
               sig_source_detect=5.0,
               min_source_area=16.,
               source_mask_size=6.,
               edges=60,
               output='Object',
               debug=False,
               showDebug=False):
    """This macro remove residual background in the cubes and fix the variance
    vector after masking sources. Sources are detected in an image created by
    collapsing the cube between min_lambda and max_lambda (considering mask_z as
    mask for bad channels). If statcube is none, it will be created and for each
    channel, the variance of the background will be used.

   Parameters
   ----------
    datacontainer : IFUcube object
        data passed through cubeClass.py
    min_lambda : int
        min channel to create the image to detected sources
    max_lambda : int
        max channel to create the image to detected sources
    mask_z : int, bool, optional
        when 1 (or True), this is a channel to be removed while
        collapsing the cube to detect sources (default is None)
    mask_xy : int, bool, optional
        when 1 (or True), this spatial pixel will be removed from
        the estimate of the b/g values (default is None)
    sig_source_detect : float
        detection sigma threshold for sources in the
        collapsed cube (default is 5.0)
    min_source_area : float
        min area for source detection in the collapsed cube (default is 16.)
    source_mask_size : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection (default is 6.)
    edges : int
        frame size removed to avoid problems related to the edge
        of the image (default is 60)
    output : string
        root file name for outputs

    Returns
    -------
    datacube, statcube : np.array
        3D data and variance cubes after residual sky subtraction and
        variance rescaled to match the background variance.
    bg_average, bg_median, bg_std, bg_var, bg_pixels : np.array
        average, median, standard deviation, variance, and number of pixels after masking
        sources, edges, and NaNs of the background.
    bg_mask_2d : np.array
        2D mask used to determine the background region. This mask has 1 if there is
        a source or is on the edge of the cube. It is 0 if the pixel is considered
        in the background estimate.
    bg_data_image : np.array
        output from statBg of the reduced background with masking applied

    """

    print("subtractBg: Starting the procedure to subtract the background")

    # Getting spectrum of the background
    bg_average, bg_median, bg_std, bg_var, \
    bg_pixels, bg_mask_2d, bg_data_image = statBg(datacontainer,
                                                  min_lambda=min_lambda,
                                                  max_lambda=max_lambda,
                                                  mask_z=mask_z,
                                                  mask_xy=mask_xy,
                                                  sig_source_detect=sig_source_detect,
                                                  min_source_area=min_source_area,
                                                  source_mask_size=source_mask_size,
                                                  edges=edges,
                                                  output=output,
                                                  debug=debug,
                                                  showDebug=showDebug)

    datacopy, statcube = datacontainer.get_data_stat()

    z_max, y_max, x_max = np.shape(datacopy)
    print("subtractBg: Subtracting background from datacube")

    for channel in range(0, z_max):
        datacopy[channel, :, :] -= bg_median[channel]
    if statcube is None:
        print("subtractBg: Creating statcube with variance inferred from background")
        statcopy = np.copy(datacopy)
        for channel in range(0, z_max):
            statcopy[channel, :, :] = bg_var[channel]
    else:
        print("subtractBg: Estimating correction for statcube variance")
        # Removing sources and edges
        statcopy_nan = np.copy(statcube)
        bg_mask_3D = np.broadcast_to((bg_mask_2d == 1), statcopy_nan.shape)
        statcopy_nan[(bg_mask_3D == 1)] = np.nan
        # Calculating average variance per channel
        # averageStatBg = np.nanmean(statcopy_nan, axis=(1, 2))  # TODO

        averageStatBg = np.nanmean(sigma_clip(statcopy_nan, maxiters=10, sigma=2.3, axis=(1, 2)), axis=(1, 2))
        del statcopy_nan
        del bg_mask_3D
        # Rescaling cube variance
        scale_factor = bg_var / averageStatBg
        statcopy = np.copy(statcube)
        for channel in range(0, z_max):
            statcopy[channel, :, :] *= scale_factor[channel]
        print("subtractBg: The average correction factor for variance is {:.5f}".format(np.average(scale_factor)))
        if debug:
            manip.nicePlot()
            plt.figure(1, figsize=(9, 6))
            plt.plot(range(0, z_max), scale_factor, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Correction Factor")
            plt.axhline(np.average(scale_factor))
            plt.savefig(output + "_VarianceCorrection.pdf", dpi=400.,
                        format="pdf", bbox_inches="tight")
            if showDebug:
                plt.show()
            plt.close()

        print("subtractBg: The average value subtracted to the b/g level is {:.5f}".format(np.average(bg_median)))
        if debug:
            manip.nicePlot()
            plt.figure(1, figsize=(9, 6))
            plt.plot(range(0, z_max), bg_median, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Median Background")
            plt.axhline(np.average(bg_median))
            plt.savefig(output + "_bgCorrection.pdf", dpi=400.,
                        format="pdf", bbox_inches="tight")
            if showDebug:
                plt.show()
            plt.close()

    gc.collect()
    return datacopy, statcopy, bg_average, bg_median, bg_std, bg_var, bg_pixels, bg_mask_2d, bg_data_image


def sources_fg(datacube,
               statcube=None,
               sig_source_detect=5.0,
               min_source_area=16.,
               source_mask_size=6.,
               max_source_size=50.,
               max_source_ell=0.9,
               rad_norm=1.,
               edges=60,
               output='Object',
               debug=False,
               showDebug=False):
    """ This macro search for sources in an image and save relevant
    information on them in a dictionary.

    Parameters
    ----------
    datacube : np.array
        data in a 2D array
    statcube : np.array
        variance in a 2D array
    sig_source_detect : float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    min_source_area : float
        min area for source detection in the collapsed
        cube. Default is 16.
    source_mask_size : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection. Default is 6.
    max_source_size : float
        sources with semi-major or semi-minor axes larger than this
        value will not considered in the foreground source model.
        Default is 50.
    max_source_ell : float
        sources with ellipticity larger than this value will not
        considered in the foreground source model. Default is 0.9.
    rad_norm : float
        radius where to normalize the sources model. Default is 1.
    edges : np.int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for outputs
    Returns
    -------
    fgSources : dict
        Dictionary containing relevant information on the detected
        sources including data, masks, flux, etc.
    fgAllSources : astropy table
        Astropy table containing the information of all sources detected
        (i.e. without any cleaning). This is the direct output of the
        source finding algorithm.
    fgBackgroundFlux, fgBackgroundSigma : float
        Average background flux and sigma after source extractions
    """

    print("sources_fg: Searching for sources in the image")
    y_max, x_max = np.shape(datacube)
    fg_xpos, fg_ypos, fg_maj, fg_min, fg_angle, fg_all_sources = find_sources(datacube, statcube,
                                                                              sig_detect=sig_source_detect,
                                                                              min_area=min_source_area)
    print("sources_fg: Detected {} sources".format(len(fg_xpos)))

    print("sources_fg: Perform some cleaning on the detected sources")
    # Checking that, within RadiusNorm, each source is at least
    # sig_source_detection/2. sigma above the background
    # skyMask is a mask where all sources are masked as 1,
    # while the background is set to 0
    fg_datacopy = np.copy(datacube)
    sky_mask = manip.location(datacube, fg_xpos, fg_ypos,
                              semi_maj=source_mask_size * fg_maj,
                              semi_min=source_mask_size * fg_min,
                              theta=fg_angle)
    fg_datacopy[(sky_mask == 1)] = np.nan

    # removing edges. This mask is 0 if it is a good pixel, 1 if it is a
    # pixel at the edge
    edges_mask = np.ones_like(sky_mask, int)
    edges_mask[int(edges):- int(edges), int(edges):- int(edges)] = 0
    fg_datacopy[(edges_mask == 1)] = np.nan

    # removing extreme values. This mask is 0 if it is a good pixel,
    # 1 if it is a pixel with an extreme value
    extreme_mask = np.ones_like(sky_mask, int)
    fg_flux = np.nanmedian(fg_datacopy)
    fg_sigma = np.nanstd(fg_datacopy)
    extreme_mask[np.abs((datacube - fg_flux) / fg_sigma) < 2.99] = 0
    fg_datacopy[(extreme_mask == 1)] = np.nan

    # Checking values of the background
    fg_flux = np.nanmedian(fg_datacopy)
    fg_sigma = np.nanstd(fg_datacopy)
    fg_data_hist, fg_data_edges = np.histogram(fg_datacopy[np.isfinite(fg_datacopy)].flatten(),
                                               bins="fd", density=True)
    # fitting of the histogram
    gauss_best, gauss_covar = curve_fit(manip.gaussian,
                                        fg_data_edges[:-1],
                                        fg_data_hist,
                                        p0=[1. / (fg_sigma * np.sqrt((2. * np.pi))),
                                            fg_flux,
                                            fg_sigma])

    # A second round of global sky subtraction is performed
    print("sources_fg: A residual background of {:.4f} counts has been removed".format(fg_flux))
    fg_data_no_bg = np.copy(datacube) - fg_flux

    # Running force photometry on the detected sources
    print("sources_fg: Aperture photometry on sources with radius {:.4f} pix.".format(rad_norm))
    fg_flux_cent, fg_err_flux_cent = manip.quickApPhotmetryNoBg(fg_data_no_bg,
                                                                statcube,
                                                                fg_xpos,
                                                                fg_ypos,
                                                                obj_rad=rad_norm)

    # Removing sources that are at the edge of the detection at the center
    fg_bright = fg_flux_cent > .5 * sig_source_detect * fg_err_flux_cent
    fg_flux_cent = fg_flux_cent[fg_bright]
    fg_err_flux_cent = fg_err_flux_cent[fg_bright]
    fg_xpos = fg_xpos[fg_bright]
    fg_ypos = fg_ypos[fg_bright]
    fg_maj = fg_maj[fg_bright]
    fg_min = fg_min[fg_bright]
    fg_angle = fg_angle[fg_bright]
    print("sources_fg: Detected {} sources above {} sigma at the center".format(len(fg_xpos), .5 * sig_source_detect))

    # Remove sources at the edge of the FOV
    fg_x_loc = (fg_xpos > edges) & (fg_xpos < (x_max - edges))
    fg_y_loc = (fg_ypos > edges) & (fg_ypos < (y_max - edges))
    fg_location = fg_x_loc & fg_y_loc
    fg_flux_cent = fg_flux_cent[fg_location]
    fg_err_flux_cent = fg_err_flux_cent[fg_location]
    fg_xpos = fg_xpos[fg_location]
    fg_ypos = fg_ypos[fg_location]
    fg_maj = fg_maj[fg_location]
    fg_min = fg_min[fg_location]
    fg_angle = fg_angle[fg_location]
    print("sources_fg: Detected {} sources after edges removal".format(len(fg_xpos)))

    # Removes crazy values axis of elliptical sources
    fg_maj_size = fg_maj < max_source_size
    fg_min_size = fg_min < max_source_size
    fg_maj_big = np.max((fg_maj, fg_min), axis=0)
    fg_maj_small = np.min((fg_maj, fg_min), axis=0)
    fg_ell = 1. - (fg_maj_small / fg_maj_big)
    fg_ell_cut = (fg_ell < np.max([np.percentile(fg_ell, 10), max_source_ell])) & (fg_ell > 0.)
    fg_shape = fg_maj_size & fg_min_size & fg_ell_cut
    fg_flux_cent = fg_flux_cent[fg_shape]
    fg_err_flux_cent = fg_err_flux_cent[fg_shape]
    fg_xpos = fg_xpos[fg_shape]
    fg_ypos = fg_ypos[fg_shape]
    fg_maj = fg_maj[fg_shape]
    fg_min = fg_min[fg_shape]
    fg_angle = fg_angle[fg_shape]
    print("sources_fg: Removing sources with size larger than {}".format(max_source_size))
    print("sources_fg: Removing sources with ellepticity larger than {}".format(
        np.max([np.percentile(fg_ell, 10), max_source_ell])))
    print("sources_fg: Detected {} sources after removing unusal shapes".format(len(fg_xpos)))

    del fg_datacopy
    del fg_bright
    del fg_location
    del fg_shape

    if debug:
        print("sources_fg: Saving degub image on {}_fgSourcesMask.pdf".format(output))

        fg_datacopy = np.copy(datacube)
        sky_mask = manip.location(datacube, fg_xpos, fg_ypos,
                                  semi_maj=source_mask_size * fg_maj,
                                  semi_min=source_mask_size * fg_min,
                                  theta=fg_angle)
        fg_datacopy[(sky_mask == 1)] = np.nan
        edges_mask = np.ones_like(sky_mask, int)
        edges_mask[int(edges):-int(edges), int(edges):-int(edges)] = 0
        fg_datacopy[(edges_mask == 1)] = np.nan

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        ax_image = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax_mask = plt.subplot2grid((1, 3), (0, 1), colspan=1)
        ax_hist = plt.subplot2grid((1, 3), (0, 2), colspan=1)

        # Plotting field image
        ax_image.imshow(datacube,
                        cmap="Greys", origin="lower",
                        vmin=fg_flux - 3. * fg_sigma,
                        vmax=fg_flux + 3. * fg_sigma)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_title(r"Collapsed image")

        # Plotting field mask
        ax_mask.imshow(fg_datacopy,
                       cmap="Greys", origin="lower",
                       vmin=fg_flux - 3. * fg_sigma,
                       vmax=fg_flux + 3. * fg_sigma)
        ax_mask.set_xlabel(r"X [Pixels]", size=30)
        ax_mask.set_ylabel(r"Y [Pixels]", size=30)
        ax_mask.set_title(r"Background")

        # Plotting pixel distribution
        ax_hist.step(fg_data_edges[:-1], fg_data_hist, color="gray",
                     zorder=3)
        ax_hist.plot(fg_data_edges[:-1], manip.gaussian(fg_data_edges[:-1], *gauss_best),
                     color='black', zorder=2)
        ax_hist.axvline(fg_flux, color="black",
                        zorder=1, linestyle=':')
        ax_hist.set_xlim(left=fg_flux - 3. * fg_sigma,
                         right=fg_flux + 3. * fg_sigma)
        ax_hist.text(0.52, 0.9, "Med b/g", transform=ax_hist.transAxes)
        ax_hist.set_ylabel(r"Pixel Distribution", size=30)
        ax_hist.set_xlabel(r"Flux", size=30)
        ax_hist.set_title(r"b/g flux distribution")
        # axHist.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        plt.tight_layout()
        plt.savefig(output + "_fgSourcesMask.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del sky_mask
        del edges_mask
        del fg_datacopy

    print("sources_fg: Filling dictionary with relevant information on the {} sources".format(len(fg_xpos)))
    # create dictionary with relevant information on the fg sources
    fg_sources = {}
    fgFlux = np.zeros_like(fg_xpos)
    fgErrFlux = np.zeros_like(fg_xpos)
    fgFluxBg = np.zeros_like(fg_xpos)

    # create elliptical mask with all sources masked
    # fgMask is a mask where all sources are masked as 1,
    # while the background is set to 0
    fg_mask = manip.location(fg_data_no_bg, fg_xpos, fg_ypos,
                             semi_maj=source_mask_size * fg_maj,
                             semi_min=source_mask_size * fg_min,
                             theta=fg_angle)

    for sourceIdx in range(0, len(fg_xpos)):
        # create elliptical mask of each source
        # fgThisSourceMask is a mask where the considered source is masked as 1,
        # while the background is set to 0
        index_mask = manip.location(fg_data_no_bg, fg_xpos[sourceIdx], fg_ypos[sourceIdx],
                                    semi_maj=source_mask_size * fg_maj[sourceIdx],
                                    semi_min=source_mask_size * fg_min[sourceIdx],
                                    theta=fg_angle[sourceIdx])
        fgThisMask, fgThisDataNoBg = np.copy(fg_mask), np.copy(fg_data_no_bg)
        # fgThisMask is a mask where all sources are masked as 1 BUT the source considered
        fgThisMask[(index_mask == 1)] = np.int(0)
        # if there is another source in the area of fgThisSourceMask
        # it will be masked within rad_norm
        for sourceIdx_tmp in range(0, len(fg_xpos)):
            if (fg_xpos[sourceIdx_tmp] != fg_xpos[sourceIdx]):
                fgContaminantSmallMask = manip.location(fg_data_no_bg, fg_xpos[sourceIdx_tmp], fg_ypos[sourceIdx_tmp],
                                                        semi_maj=rad_norm,
                                                        semi_min=rad_norm,
                                                        theta=fg_angle[sourceIdx_tmp])
                fgThisMask[(fgContaminantSmallMask == 1)] = 1
        # Loading only data from the current source
        fgThisSourceData = np.copy(datacube)
        fgThisSourceData[(index_mask == 0)] = 0.
        fgThisSourceDataNoBg = np.copy(fg_data_no_bg)
        fgThisSourceDataNoBg[(index_mask == 0)] = 0.
        fgThisSourceDataNoBg[(fgThisMask == 1)] = 0.

        # Loading dictionary with relevant informations
        fg_sources[sourceIdx] = {}
        fg_sources[sourceIdx]["x"] = fg_xpos[sourceIdx]
        fg_sources[sourceIdx]["y"] = fg_ypos[sourceIdx]
        fg_sources[sourceIdx]["a"] = fg_maj[sourceIdx]
        fg_sources[sourceIdx]["b"] = fg_min[sourceIdx]
        fg_sources[sourceIdx]["theta"] = fg_angle[sourceIdx]
        fg_sources[sourceIdx]["flux"] = fg_flux_cent[sourceIdx]
        fg_sources[sourceIdx]["errFlux"] = fg_err_flux_cent[sourceIdx]
        fg_sources[sourceIdx]["radiusFlux"] = rad_norm
        fg_sources[sourceIdx]["fluxBg"] = fg_flux
        fg_sources[sourceIdx]["errFluxBg"] = fg_sigma
        fg_sources[sourceIdx]["sourceMask"] = index_mask
        fg_sources[sourceIdx]["contaminantMask"] = fgThisMask
        fg_sources[sourceIdx]["sourceData"] = fgThisSourceData
        fg_sources[sourceIdx]["sourceDataNoBg"] = fgThisSourceDataNoBg

    """
    # ToDo : this is not working at the moment.
    print("sources_fg: Saving foreground source dictionary in {}".format(output+"_fgSources.json"))
    with open(output+"_fgSources.json", "w") as jfile:
        fgSourcesJason = muutils.jsonify(fgSources)
        json.dump(fgSourcesJason, jfile, sort_keys=True, indent=4, 
                  separators=(',', ': '), easy_to_read=True, overwrite=True)
    """

    # Deleting temporary images to clear up memory
    del fgThisSourceDataNoBg
    del fgThisSourceData
    del fgThisMask
    del index_mask

    return fg_sources, fg_all_sources, fg_flux, fg_sigma


def clean_fg(datacontainer,
             min_lambda=None,
             max_lambda=None,
             mask_z=None,
             mask_x=None,
             mask_y=None,
             mask_xy_rad=None,
             sig_source_detect=5.0,
             min_source_area=16.,
             mask_size=6.,
             max_source_size=50.,
             max_source_ell=0.9,
             bg_source=False,
             rad_norm=1.,
             edges=60,
             output='Object',
             debug=False,
             showDebug=False,
             deepDebug=False):
    """ This macro removes sources from a cube creating a model from the data
    itself. The model will be created collapsing the cube between min_lambda
    and max_lambda (considering mask_z as mask for bad channels). The macro
    will search for sources in this image, and create a normalized model for
    each of them.
    Finally, this model will be propagated in the entire cube and then removed.

    Parameters
    ----------
    datacontainer : IFUcube object
        data intialized in cubeClass.py
    min_lambda : int
        min channel to create the image where to detect sources
    max_lambda : int
        max channel to create the image where to detect sources
    mask_z : np.array
        when 1 (or True), this is a channel to be removed
    mask_x, mask_y, mask_xy_rad : floats
        x,y location of a source NOT to be subtracted as f/g model.
        Any sources located within mask_xy_rad from mask_x, maskY will
        not be considered in the creation of the model.
    sig_source_detect : float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    min_source_area : float
        min area for source detection in the collapsed
        cube. Default is 16.
    mask_size : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection. Default is 6.
    max_source_size : float
        sources with semi-major or semi-minor axes larger than this
        value will not be considered in the foreground source model.
        Default is 50.
    max_source_ell : float
        sources with ellipticity larger than this value will not be
        considered in the foreground source model. Default is 0.9.
    bg_source : bool
        if True, an additional local background subtraction will be
        performed around each detected sources. Default is True
    rad_norm : float
        radius where to normalize the sources model. Default is 1.
    edges : int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for outputs

    Returns
    -------
    dataCubeClean : np.array
        data cube from which the detected sources have been removed
    fgSources : dict
        Dictionary containing relevant information on the detected
        sources including data, masks, flux, etc.
    """

    print("clean_fg: Starting the cleaning of the cube")
    datacopy, statcopy = datacontainer.get_data_stat()
    z_max, y_max, x_max = np.shape(datacopy)

    print("clean_fg: Collapsing cube")
    fg_data, fg_stat = manip.collapse_container(datacontainer, mask_z=mask_z,
                                                min_lambda=min_lambda, max_lambda=max_lambda)

    print("clean_fg: Searching for sources in the collapsed cube")
    fg_sources, fg_all_sources, fg_bg_flux, fg_bg_sigma = sources_fg(fg_data, statcube=fg_stat,
                                                                     sig_source_detect=sig_source_detect,
                                                                     min_source_area=min_source_area,
                                                                     source_mask_size=mask_size,
                                                                     max_source_size=max_source_size,
                                                                     max_source_ell=max_source_ell,
                                                                     rad_norm=rad_norm,
                                                                     edges=edges,
                                                                     output=output,
                                                                     debug=debug)

    print("clean_fg: Subtracting contaminants")
    # Subtracting the f/g contamination
    datacopy_clean = np.copy(datacopy)
    datacopy_model = np.zeros_like(datacopy)
    for sourceIdx in range(0, len(fg_sources)):
        from IPython import embed
        embed()
        print("clean_fg: Removing source {}".format(sourceIdx))
        # creating normalized model
        fg_model = (fg_sources[sourceIdx]["sourceDataNoBg"] / fg_sources[sourceIdx]["flux"])
        fg_model[(fg_sources[sourceIdx]["sourceMask"] == 0)] = 0.
        print("         The min, max values for fg model are: {:03.4f}, {:03.4f}".format(np.min(fg_model),
                                                                                         np.max(fg_model)))
        # extract spectrum of the source from the datacube
        fg_source_extent = np.max([fg_sources[sourceIdx]["a"], fg_sources[sourceIdx]["b"]])
        bg_inner_rad = 1.1 * mask_size * (fg_source_extent + fg_sources[sourceIdx]["radiusFlux"])
        bg_outer_rad = bg_inner_rad + (5. * fg_sources[sourceIdx]["radiusFlux"])
        if bg_source:
            fg_flux_source, fg_err_flux_source, \
            fg_bg_flux_source = manip.quickSpectrum(datacopy, statcopy=statcopy,
                                                    x_pos=fg_sources["x"][sourceIdx],
                                                    y_pos=fg_sources["y"][sourceIdx],
                                                    radius_pos=fg_sources["radiusFlux"][sourceIdx],
                                                    inner_rad=bg_inner_rad,
                                                    outer_rad=bg_outer_rad,
                                                    void_mask=fg_sources["contaminantMask"][sourceIdx])
        else:
            fg_flux_source, fg_err_flux_source = manip.quickSpectrumNoBg(datacopy, statcube=statcopy,
                                                                         x_pos=fg_sources["x"][sourceIdx],
                                                                         y_pos=fg_sources["y"][sourceIdx],
                                                                         radius_pos=fg_sources[sourceIdx]["radiusFlux"])
            fg_bg_flux_source = None
        keep_source = True
        if mask_xy_rad is not None:
            dist_from_masked = manip.distFromPixel(0., mask_y, mask_x, 0., np.array(fg_sources[sourceIdx]["y"]),
                                                   np.array(fg_sources[sourceIdx]["x"]))
            if dist_from_masked < mask_xy_rad:
                keep_source = False
                print("cleanFg: Source not removed")
                print("         it is located {}<{} pixel away from the XYmask".format(dist_from_masked, mask_xy_rad))

        if keep_source:
            for channel in range(0, z_max):
                # selecting only where the source is significantly detected
                if fg_flux_source[channel] > 0.5 * fg_err_flux_source[channel]:
                    if bg_source:
                        datacopy_clean[channel, :, :] -= ((fg_model * fg_flux_source[channel]) + (
                                fg_sources[sourceIdx]["sourceMask"] * fg_bg_flux_source[channel]))
                        datacopy_model[channel, :, :] += ((fg_model * fg_flux_source[channel]) + (
                                fg_sources[sourceIdx]["sourceMask"] * fg_bg_flux_source[channel]))
                    else:
                        datacopy_clean[channel, :, :] -= (fg_model * fg_flux_source[channel])
                        datacopy_model[channel, :, :] += (fg_model * fg_flux_source[channel])

        if deepDebug:
            print("clean_fg: Spectrum of the source {}".format(sourceIdx))

            plt.figure(1, figsize=(18, 6))
            gs = gridspec.GridSpec(1, 3)

            ax_imag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
            ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

            temp_fg_model_xmin, temp_fg_model_xmax = int(fg_sources[sourceIdx]["x"] - bg_inner_rad), int(
                fg_sources[sourceIdx]["x"] + bg_inner_rad)
            temp_fg_model_ymin, temp_fg_model_ymax = int(fg_sources[sourceIdx]["y"] - bg_inner_rad), int(
                fg_sources[sourceIdx]["y"] + bg_inner_rad)
            temp_fg_model = np.copy(
                fg_model[temp_fg_model_ymin:temp_fg_model_ymax, temp_fg_model_xmin:temp_fg_model_xmax])
            ax_imag.imshow(temp_fg_model,
                           cmap="Greys", origin="lower",
                           vmin=0.,
                           vmax=1. / (np.pi * rad_norm * rad_norm))
            ax_imag.set_xlabel(r"X [Pixels]", size=30)
            ax_imag.set_ylabel(r"Y [Pixels]", size=30)

            ax_spec.plot(fg_flux_source, color='black', zorder=3, label='Flux')
            ax_spec.plot(fg_err_flux_source, color='gray', alpha=0.5, zorder=2, label='Error')
            if bg_source:
                ax_spec.plot(fg_bg_flux_source * (np.pi * rad_norm * rad_norm), color='red', alpha=0.5, zorder=1,
                             label='b/g')
            ax_spec.legend()
            ax_spec.set_xlabel(r"Channel", size=30)
            ax_spec.set_ylabel(r"Flux", size=30)

            plt.tight_layout()
            plt.show()
            plt.close()
            del temp_fg_model

    print("clean_fg: Source cleaning performed")

    if debug:
        print("clean_fg: Saving debug image on {}_fgSourcesCleaned.pdf".format(output))
        fg_data_clean = manip.collapse_cube(datacopy_clean, mask_z=mask_z,
                                            min_lambda=min_lambda, max_lambda=max_lambda)
        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_clean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        ax_image.imshow(fg_data,
                        cmap="Greys", origin="lower",
                        vmin=fg_bg_flux - 3. * fg_bg_sigma,
                        vmax=fg_bg_flux + 3. * fg_bg_sigma)
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_title(r"Collapsed image")

        for sourceIdx in range(0, len(fg_sources)):
            fg_source_artist = Ellipse(xy=(fg_sources[sourceIdx]["x"], fg_sources[sourceIdx]["y"]),
                                       width=mask_size * fg_sources[sourceIdx]["a"],
                                       height=mask_size * fg_sources[sourceIdx]["b"],
                                       angle=fg_sources[sourceIdx]["theta"])
            fg_source_artist.set_facecolor("none")
            fg_source_artist.set_edgecolor("red")
            fg_source_artist.set_alpha(0.8)
            ax_image.add_artist(fg_source_artist)

        # Plotting cleaned image
        ax_clean.imshow(fg_data_clean,
                        cmap="Greys", origin="lower",
                        vmin=fg_bg_flux - 3. * fg_bg_sigma,
                        vmax=fg_bg_flux + 3. * fg_bg_sigma)
        ax_clean.set_xlabel(r"X [Pixels]", size=30)
        ax_clean.set_ylabel(r"Y [Pixels]", size=30)
        ax_clean.set_title(r"Cleaned image")

        plt.tight_layout()
        plt.savefig(output + "_fgSourcesCleaned.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()

    print("clean_fg: Saving source list on {}_fgSources.txt".format(output))
    f = open(output + "_fgSources.txt", 'w')
    f.write("Idx fgXPix fgYPix fgAPix fgBPix fgTheta\n")
    for sourceIdx in range(0, len(fg_sources)):
        f.write("{:.0f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(sourceIdx, fg_sources[sourceIdx]["x"],
                                                                     fg_sources[sourceIdx]["y"],
                                                                     fg_sources[sourceIdx]["a"],
                                                                     fg_sources[sourceIdx]["b"],
                                                                     fg_sources[sourceIdx]["theta"]))
    f.close()

    gc.collect()
    return datacopy_clean, datacopy_model, fg_sources


def makePsf(datacontainer,
            x_pos,
            y_pos,
            statcube=None,
            min_lambda=None,
            max_lambda=None,
            maskZ=None,
            radius_pos=2.,
            inner_rad=10.,
            outer_rad=15.,
            rad_psf=50.,
            cType="sum",
            norm=True,
            debug=False,
            showDebug=False):
    """Given an IFUcube, or 3D data and variance arrays, the macro collapses
    it's data and variance along the z-axis between min_lambda and
    max_lambda. If mask_z is given, channels masked as 1 (or True) are removed.
    if cType is set to 'average', the macro uses the stat information to perform a
    weighted mean along the velocity axis. In other words, each spaxel of the resulting
    image will be the weighted mean spectrum of that spaxels along the wavelengths.
    If norm is 'True' the macro normalize the flux of the PSF within radius_pos = 1.

    Parameters
    ----------
    datacontainer : IFUcube Object, np.array
        data initiated in cubeClass.py, or 3D data array
    x_pos : float
        x-location of the source in pixel
    y_pos : float
        y-location of the source in pixel
    statcube : np.array, optional
        variance data array, optional if read in IFUcube for datacube
    min_lambda : int, optional
        min channel to create collapsed image (default is None)
    max_lambda : int, optional
        max channel to create collapsed image (default is None)
    maskZ
        when 1 (or True), this is a channel to be removed (default is None)
    radius_pos : float
        radius where to perform the aperture photometry (default is 2.)
    inner_rad : float
        inner radius of the background region in pixel (default is 10.)
    outer_rad : float
        outer radius of the background region in pixel (default is 15.)
    rad_psf : float
        radius of the PSF image to be created. Outside
        these pixels values are set to zero (default is 50.)
    cType : str
        type of combination for PSF creation:
        'average' is weighted average
        'sum' is direct sum of all pixels
    norm : bool
        if 'True' normalizes the central regions of the
        PSF to 1 (default is True)

    Returns
    -------
    psfData, psfStat : np.array
        PSF data and variance images
    """
    print("makePsf: Creating PSF model")

    if statcube is None:
        if cType == 'sum':
            print("makePsf: Summing channels")
            psf_data, psf_stat = manip.collapse_cube(datacontainer, min_lambda, max_lambda, mask_z=maskZ)
        else:
            print("makePsf: Average combining channels")
            psf_data, psf_stat = manip.collapse_mean_container(datacontainer, min_lambda, max_lambda)
    else:
        datacopy = np.copy(datacontainer)
        if cType == 'sum':
            print("makePsf: Summing channels")
            psf_data = manip.collapse_cube(datacopy, min_lambda, max_lambda, mask_z=maskZ)
            psf_stat = manip.collapse_cube(statcube, min_lambda, max_lambda, mask_z=maskZ)
        else:
            print("makePsf: Average combining channels")
            psf_data, psf_stat = manip.collapse_mean_cube(datacontainer, statcube, min_lambda, max_lambda)

    psf_flux, psf_err_flux, psf_ave_flux = manip.quickApPhotmetry(psf_data, psf_stat, x_pos=x_pos, y_pos=y_pos,
                                                                  radius_pos=radius_pos, inner_rad=inner_rad,
                                                                  outer_rad=outer_rad)

    print("makePsf: Removing local background of {}".format(psf_ave_flux))
    psf_data = psf_data - psf_ave_flux

    if norm:
        print("makePsf: Normalizing central region to 1")
        print("         (i.e. correcting for a factor {}".format(psf_flux))
        psf_norm = psf_flux
    else:
        psf_norm = 1.

    psf_data, psf_stat = psf_data / psf_norm, psf_stat / (psf_norm ** 2.)

    print("makePsf: Creating circular mask around the position {}, {}".format(x_pos, y_pos))
    psf_mask = manip.location(psf_data, x_position=x_pos, y_position=y_pos,
                              semi_maj=rad_psf, semi_min=rad_psf)
    psf_data[(psf_mask == 0)] = 0.
    psf_stat[(psf_mask == 0)] = 0.

    if debug:
        print("makePsf: Creating debug images")

        manip.nicePlot()

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        ax_image = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        ax_stat = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        ax_image.imshow(psf_data,
                        cmap="Greys", origin="lower",
                        vmin=0.,
                        vmax=0.3 * np.nanmax(psf_data))
        ax_image.set_xlabel(r"X [Pixels]", size=30)
        ax_image.set_ylabel(r"Y [Pixels]", size=30)
        ax_image.set_xlim(left=x_pos - rad_psf, right=x_pos + rad_psf)
        ax_image.set_ylim(bottom=y_pos - rad_psf, top=y_pos + rad_psf)
        ax_image.set_title(r"PSF model")

        inner_artist = Ellipse(xy=(x_pos, y_pos),
                               width=inner_rad,
                               height=inner_rad,
                               angle=0.)
        inner_artist.set_facecolor("none")
        inner_artist.set_edgecolor("red")
        inner_artist.set_alpha(0.8)
        ax_image.add_artist(inner_artist)

        outer_artist = Ellipse(xy=(x_pos, y_pos),
                               width=outer_rad,
                               height=outer_rad,
                               angle=0.)
        outer_artist.set_facecolor("none")
        outer_artist.set_edgecolor("red")
        outer_artist.set_alpha(0.8)
        ax_image.add_artist(outer_artist)

        # Plotting cleaned image
        x_plot_min = int(x_pos - inner_rad)
        x_plot_max = int(x_pos + inner_rad)
        y_plot_min = int(y_pos - inner_rad)
        y_plot_max = int(y_pos + inner_rad)

        ax_stat.imshow(psf_stat,
                       cmap="Greys", origin="lower",
                       vmin=np.nanmin(psf_stat[y_plot_min:y_plot_max,
                                      x_plot_min:x_plot_max]),
                       vmax=0.5 * np.nanmax(psf_stat[y_plot_min:y_plot_max,
                                            x_plot_min:x_plot_max]))

        ax_stat.set_xlabel(r"X [Pixels]", size=30)
        ax_stat.set_ylabel(r"Y [Pixels]", size=30)
        ax_stat.set_xlim(left=x_pos - rad_psf, right=x_pos + rad_psf)
        ax_stat.set_ylim(bottom=y_pos - rad_psf, top=y_pos + rad_psf)
        ax_stat.set_title(r"PSF Variance")

        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()

    gc.collect()
    return psf_data, psf_stat


def clean_psf(datacontainer,
              psf_model,
              x_pos,
              y_pos,
              radius_pos=2.,
              inner_rad=10.,
              outer_rad=15.,
              bg_psf=True,
              debug=False,
              show_debug=False):
    """Given a cube and a PSF model, the macro subtracts the PSF contribution
    along the wavelength axis. It assumes that the PSF model is normalized to
    one within rad_psf and that the PSF model is centered in the same location
    of the object you want to remove. This will be improved in the future.

    Parameters
    ----------
    datacontainer : IFUcube object
        Data initialized in cubeClass.py
    psf_model : np.array
        array from running makePsf to be reduced in this function
    x_pos : float
        x-location of the source in pixel
    y_pos : float
        y-location of the source in pixel
    radius_pos : float
        radius where to perform the aperture photometry
        to remove the PSF contribution (default is 2.)
    inner_rad : float
        inner radius of the background region in pixel (default is 10.)
    outer_rad : float
        outer radius of the background region in pixel (default is 15.)
    bg_psf : bool
        if True, an additional local background subtraction will be
        performed around the source (default is True)

    Returns
    -------
    psfSubCube, psfModel : np.array
        PSF subtracted cube and PSF model cube
    """

    print("clean_psf: PSF subtraction on cube")
    datacopy, statcopy = datacontainer.get_data_stat()
    datacopy_model = np.zeros_like(datacopy)
    z_max, y_max, x_max = np.shape(datacopy)

    print("clean_psf: The min, max values for PSF model are: {:03.4f}, {:03.4f}".format(np.min(psf_model),
                                                                                       np.max(psf_model)))
    # extract spectrum of the source from the datacube
    if bg_psf:
        flux_source, err_flux_source, bg_flux_source = manip.quickSpectrum(datacontainer=datacopy,
                                                                           statcube=statcopy,
                                                                           x_pos=x_pos, y_pos=y_pos,
                                                                           radius_pos=radius_pos,
                                                                           inner_rad=inner_rad,
                                                                           outer_rad=outer_rad)
    else:
        flux_source, err_flux_source = manip.quickSpectrumNoBg(datacontainer=datacopy,
                                                               statcube=statcopy,
                                                               x_pos=x_pos, y_pos=y_pos,
                                                               radius_pos=radius_pos)
        bg_flux_source = np.zeros_like(flux_source)

    for channel in range(0, z_max):
        # selecting only where the source is significantly detected
        if flux_source[channel] > 0.5 * (err_flux_source[channel]):
            datacopy[channel, :, :] -= ((psf_model * flux_source[channel]) + bg_flux_source[channel])
            datacopy_model[channel, :, :] += ((psf_model * flux_source[channel]) + bg_flux_source[channel])

    if debug:
        print("clean_psf: Spectrum of the source")

        manip.nicePlot()

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        ax_imag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        ax_spec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

        model_temp_x_min, model_temp_x_max = int(x_pos - inner_rad), int(x_pos + inner_rad)
        model_temp_y_min, model_temp_y_max = int(y_pos - inner_rad), int(y_pos + inner_rad)
        model_temp = np.copy(psf_model[model_temp_y_min:model_temp_y_max, model_temp_x_min:model_temp_x_max])
        ax_imag.imshow(model_temp,
                       cmap="Greys", origin="lower",
                       vmin=0.,
                       vmax=1. / (np.pi * radius_pos * radius_pos))
        ax_imag.set_xlabel(r"X [Pixels]", size=30)
        ax_imag.set_ylabel(r"Y [Pixels]", size=30)

        ax_spec.plot(flux_source, color='black', zorder=3, label='Flux')
        ax_spec.plot(err_flux_source, color='gray', alpha=0.5, zorder=2, label='Error')
        if bg_psf:
            ax_spec.plot(bg_flux_source * (np.pi * radius_pos * radius_pos), color='red', alpha=0.5, zorder=1,
                         label='b/g')
        ax_spec.legend()
        ax_spec.set_xlabel(r"Channel", size=30)
        ax_spec.set_ylabel(r"Flux", size=30)

        plt.tight_layout()
        if show_debug:
            plt.show()
        plt.close()
        del model_temp

    print("clean_psf: PSF cleaning performed")

    gc.collect()
    return datacopy, datacopy_model
