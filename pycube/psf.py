import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import gridspec

import numpy as np
from scipy.optimize import curve_fit

from pycube.core import manip
from pycube.core import background

import sep
import gc


def find_sources(datacontainer, statcube = None,
                 min_lambda=None, max_lambda=None,
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
    datacontainer : np.array
            data cube. 2D or 3D
    min_lambda : float
        minimum wavelength value to collapse 3D image (default is None)
    max_lambda : float
        maximum wavelength value to collapse 3D image (default is None)
    var_factor : int, float, optional
        affects generated variance, if variance is auto-generated from image data (default 5.)
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
    xPix : np.array
    yPix : np.array
    aPix : np.array
    bPix : np.array
    angle : np.array
    all_objects : np.array
    """
    if statcube is None:
        datacopy, statcopy = datacontainer.get_data_stat()
    else:
        datacopy = datacontainer
        statcopy = statcube
    data_background, var_background = manip.collapse_container(datacontainer, min_lambda=min_lambda,
                                                               max_lambda=max_lambda, var_thresh=var_factor)

    image_background = background.sextractor_background(data_background, var_background, threshold)
    void_background = data_background - image_background

    # print("find_sources: Searching sources {}-sigma above noise".format(sig_detect))
    all_objects = sep.extract(void_background, sig_detect,
                              err=var_background,
                              minarea=min_area,
                              filter_type='matched',
                              gain=gain,
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
                    sig_source_detect=5.0,
                    min_source_area=16.,
                    source_mask_size=6.,
                    source_size_max=50.,
                    source_ellipse_max=0.9,
                    edges=10):
    """

    Parameters
    ----------
    datacontainer:

    sig_source_detect:

    min_source_area:
    source_mask_size:
    source_size_max:
    source_ellipse_max:
    edges:

    Returns
    -------

    """
    datacopy, statcopy = datacontainer.get_data_stat()
    z_max, y_max, x_max = np.shape(datacopy)
    cube_bg = np.full_like(datacopy, np.nan)
    mask_bg = np.copy(cube_bg)
    for index in range(z_max):
        tmpdatacopy = np.copy(datacopy[index, :, :])
        tmpstatcopy = np.copy(statcopy[index, :, :])
        x_pos, y_pos, maj_axis, min_axis, angle, all_objects = find_sources(datacontainer, min_lambda=index,
                                                                            max_lambda=index,
                                                                            sig_detect=sig_source_detect,
                                                                            min_area=min_source_area)
        maskBg2D = np.zeros_like(tmpdatacopy)
        sky_mask = manip.location(tmpdatacopy,
                                  x_position=x_pos, y_position=y_pos,
                                  semi_maj=source_mask_size * maj_axis,
                                  semi_min=source_mask_size * min_axis,
                                  theta=angle)

        maskBg2D[(sky_mask == 1)] = 1

        edges_mask = np.ones_like(sky_mask, int)
        edges_mask[int(edges):-int(edges), int(edges):-int(edges)] = 0
        maskBg2D[(edges_mask == 1)] = 1

        mask_Bg_3D = np.broadcast_to((maskBg2D == 1), tmpdatacopy.shape)

        tmpdatacopy[(mask_Bg_3D == 1)] = np.nan
        bgDataImage = np.copy(tmpdatacopy)
        bgDataImage[(maskBg2D == 1)] = np.nan
        del tmpdatacopy
        del tmpstatcopy
        cube_bg[index] = bgDataImage
        del bgDataImage
        mask_bg[index] = maskBg2D
        del maskBg2D
    return cube_bg, mask_bg


def statBg(datacontainer,
           min_lambda=None,
           max_lambda=None,
           maskZ=None,
           maskXY=None,
           sigSourceDetection=5.0,
           minSourceArea=16.,
           sizeSourceMask=6.,
           maxSourceSize=50.,
           maxSourceEll=0.9,
           edges=60,
           output='Object',
           debug=False,
           showDebug=False):
    """This estimates the sky background of a MUSE cube after removing sources.
    Sources are detected in an image created by collapsing the cube between minChannel
    and maxChannel (considering maskZ as mask for bad channels).
    Average, std, and median will be saved.

    Parameters
    ----------
    datacontainer : file object
        currently specific to .fits file objects
    min_lambda : int
        min channel to create the image where to detect sources
    max_lambda : int
        max channel to create the image where to detect sources
    maskZ : int, bool, optional
        when 1 (or True), this is a channel to be removed (default is None)
    maskXY : int, bool, optional
        when 1 (or True), this spatial pixel will remove from
        the estimate of the b/g values (default is None)
    sigSourceDetection : float
        detection sigma threshold for sources in the
        collapsed cube (default is 5.0)
    minSourceArea : float
        min area for source detection in the collapsed
        cube (default is 16)
    sizeSourceMask : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection. Default is 6.
    maxSourceSize : float
        sources with semi-major or semi-minor axes larger than this
        value will not be considered in the foreground source model (default is 50)
    maxSourceEll : float
        sources with ellipticity larger than this value will not be
        considered in the foreground source model (default is 0.9)
    edges : int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for output

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

    print("statBg: Collapsing cube")

    data_background, _ = manip.collapse_container(datacontainer, min_lambda=min_lambda,
                                                  max_lambda=max_lambda)
    print("statBg: Searching for sources in the collapsed cube")
    x_pos, y_pos, maj_axis, min_axis, angle, all_objects = find_sources(datacontainer,
                                                                        sig_detect=sigSourceDetection,
                                                                        min_area=minSourceArea)

    print("statBg: Detected {} sources".format(len(x_pos)))
    print("statBg: Masking sources")
    maskBg2D = np.zeros_like(data_background)

    sky_mask = manip.location(data_background, x_position=x_pos, y_position=y_pos,
                              semi_maj=sizeSourceMask * maj_axis,
                              semi_min=sizeSourceMask * min_axis,
                              theta=angle)

    maskBg2D[(sky_mask == 1)] = 1

    print("statBg: Masking Edges")
    # removing edges. This mask is 0 if it is a good pixel, 1 if it is a
    # pixel at the edge
    edges_mask = np.ones_like(sky_mask, dtype=int)
    edges_mask[int(edges):-int(edges), int(edges):-int(edges)] = 0
    maskBg2D[(edges_mask == 1)] = 1

    if maskXY is not None:
        print("statBg: Masking spatial pixels from input maskXY")
        maskBg2D[(maskXY == 1)] = 1

    print("statBg: Performing b/g statistic")
    datacopy = datacontainer.get_data()
    mask_Bg_3D = np.broadcast_to((maskBg2D == 1), datacopy.shape)
    datacopy[(mask_Bg_3D == 1)] = np.nan
    averageBg, stdBg, medianBg, varBg, pixelsBg = np.nanmean(datacopy, axis=(1, 2)), \
                                                  np.nanstd(datacopy, axis=(1, 2)), \
                                                  np.nanmedian(datacopy, axis=(1, 2)), \
                                                  np.nanvar(datacopy, axis=(1, 2)), \
                                                  np.count_nonzero(~np.isnan(datacopy), axis=(1, 2))
    bgDataImage = np.copy(data_background)
    bgDataImage[(maskBg2D == 1)] = np.nan

    if debug:
        print("statBg: Saving debug image on {}_BgRegion.pdf".format(output))
        tempBgFlux = np.nanmean(bgDataImage)
        tempBgStd = np.nanstd(bgDataImage)

        plt.figure(1, figsize=(12, 6))

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axClean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        axImage.imshow(data_background,
                       cmap="Greys", origin="lower",
                       vmin=tempBgFlux - 3. * tempBgStd,
                       vmax=tempBgFlux + 3. * tempBgStd)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_title(r"Collapsed image")

        # Plotting background image
        axClean.imshow(bgDataImage,
                       cmap="Greys", origin="lower",
                       vmin=tempBgFlux - 3. * tempBgStd,
                       vmax=tempBgFlux + 3. * tempBgStd)
        axClean.set_xlabel(r"X [Pixels]", size=30)
        axClean.set_ylabel(r"Y [Pixels]", size=30)
        axClean.set_title(r"b/g image")

        plt.tight_layout()
        plt.savefig(output + "_BgRegion.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()

    # preserve memory
    del data_background
    del mask_Bg_3D
    del sky_mask
    del edges_mask
    del all_objects
    del datacopy
    gc.collect()

    return averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D, bgDataImage


def subtractBg(datacontainer,
               min_lambda=None,
               max_lambda=None,
               maskZ=None,
               maskXY=None,
               sigSourceDetection=5.0,
               minSourceArea=16.,
               sizeSourceMask=6.,
               maxSourceSize=50.,
               maxSourceEll=0.9,
               edges=60,
               output='Object',
               debug=False,
               showDebug=False):
    """This macro remove residual background in the cubes and fix the variance
    vector after masking sources. Sources are detected in an image created by
    collapsing the cube between min_lambda and max_lambda (considering maskZ as
    mask for bad channels). If statCube is none, it will be created and for each
    channel, the variance of the background will be used.

   Parameters
   ----------
    datacontainer : IFUcube object
        data passed through cubeClass.py
    min_lambda : int
        min channel to create the image to detected sources
    max_lambda : int
        max channel to create the image to detected sources
    maskZ : int, bool, optional
        when 1 (or True), this is a channel to be removed while
        collapsing the cube to detect sources (default is None)
    maskXY : int, bool, optional
        when 1 (or True), this spatial pixel will be removed from
        the estimate of the b/g values (default is None)
    sigSourceDetection : float
        detection sigma threshold for sources in the
        collapsed cube (default is 5.0)
    minSourceArea : float
        min area for source detection in the collapsed cube (default is 16.)
    sizeSourceMask : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection (default is 6.)
    maxSourceSize : float
        sources with semi-major or semi-minor axes larger than this
        value will not be considered in the foreground source model (default is 50.)
    maxSourceEll : float
        sources with ellipticity larger than this value will not
        be considered in the foreground source model (default is 0.9)
    edges : int
        frame size removed to avoid problems related to the edge
        of the image (default is 60)
    output : string
        root file name for outputs

    Returns
    -------
    datacopy, statcopy : np.array
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
    bg_average, bg_median, bg_std, bg_var, bg_pixels, bg_mask_2d, bg_data_image = statBg(datacontainer,
                                                                                min_lambda=min_lambda,
                                                                                max_lambda=max_lambda,
                                                                                maskZ=maskZ,
                                                                                maskXY=maskXY,
                                                                                sigSourceDetection=sigSourceDetection,
                                                                                minSourceArea=minSourceArea,
                                                                                sizeSourceMask=sizeSourceMask,
                                                                                maxSourceSize=maxSourceSize,
                                                                                maxSourceEll=maxSourceEll,
                                                                                edges=edges,
                                                                                output=output,
                                                                                debug=debug,
                                                                                showDebug=showDebug)

    datacopy, statcube = datacontainer.get_data_stat()

    z_max, y_max, x_max = datacontainer.get_dimensions()
    print("subtractBg: Subtracting background from dataCube")

    for channel in range(0, z_max):
        datacopy[channel, :, :] -= bg_median[channel]
    if statcube is None:
        print("subtractBg: Creating statCube with variance inferred from background")
        statcopy = np.copy(datacontainer)
        for channel in range(0, z_max):
            statcopy[channel, :, :] = bg_var[channel]
    else:
        print("subtractBg: Estimating correction for statCube variance")
        # Removing sources and edges
        statcopy_Nan = np.copy(statcube)
        maskBg3D = np.broadcast_to((bg_mask_2d == 1), statcopy_Nan.shape)
        statcopy_Nan[(maskBg3D == 1)] = np.nan
        # Calculating average variance per channel
        averageStatBg = np.nanmean(statcopy_Nan, axis=(1, 2))
        del statcopy_Nan
        del maskBg3D
        # Rescaling cube variance
        scaleFactor = bg_var / averageStatBg
        statcopy = np.copy(statcube)
        for channel in range(0, z_max):
            statcopy[channel, :, :] *= scaleFactor[channel]
        print("subtractBg: The average correction factor for variance is {:.5f}".format(np.average(scaleFactor)))
        if debug:
            manip.nicePlot()
            plt.figure(1, figsize=(9, 6))
            plt.plot(range(0, z_max), scaleFactor, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Correction Factor")
            plt.axhline(np.average(scaleFactor))
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


def sourcesFg(dataImage,
              statImage=None,
              sigSourceDetection=5.0,
              minSourceArea=16.,
              sizeSourceMask=6.,
              maxSourceSize=50.,
              maxSourceEll=0.9,
              radiusNorm=1.,
              edges=60,
              output='Object',
              debug=False,
              showDebug=False):
    """ This macro search for sources in an image and save relevant
    information on them in a dictionary.

    Parameters
    ----------
    dataImage : np.array
        data in a 2D array
    statImage : np.array
        variance in a 2D array
    sigSourceDetection : float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    minSourceArea : float
        min area for source detection in the collapsed
        cube. Default is 16.
    sizeSourceMask : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection. Default is 6.
    maxSourceSize : float
        sources with semi-major or semi-minor axes larger than this
        value will not considered in the foreground source model.
        Default is 50.
    maxSourceEll : float
        sources with ellipticity larger than this value will not
        considered in the foreground source model. Default is 0.9.
    radiusNorm : np.float
        radius where to normalize the sources model. Default is 1.
    edges : np.int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for outputs
    Returns
    -------
    fgSources : dict
        Dictionary containing relvant informations on the detected
        sources including data, masks, flux, etc.
    fgAllSources : astropy table
        Astropy table containing the information of all sources detected
        (i.e. without any cleaning). This is the direct output of the
        source finding algorithm.
    fgBackgroundFlux, fgBackgroundSigma : np.float
        Average background flux and sigma after source extractions
    """

    print("sourcesFg: Searching for sources in the image")
    yMax, xMax = np.shape(dataImage)
    fgXPix, fgYPix, fgAPix, fgBPix, fgAngle, fgAllSources = find_sources(dataImage, statImage,
                                                                         sig_detect=sigSourceDetection,
                                                                         min_area=minSourceArea)
    print("sourcesFg: Detected {} sources".format(len(fgXPix)))

    print("sourcesFg: Perform some cleaning on the detected sources")
    # Checking that, within RadiusNorm, each source is at least
    # sig_source_detection/2. sigma above the background
    # skyMask is a mask where all sources are masked as 1,
    # while the background is set to 0
    FgDataBackground = np.copy(dataImage)
    skyMask = manip.location(dataImage, fgXPix, fgYPix,
                             semi_maj=sizeSourceMask * fgAPix,
                             semi_min=sizeSourceMask * fgBPix,
                             theta=fgAngle)
    FgDataBackground[(skyMask==1)] = np.nan

    # removing edges. This mask is 0 if it is a good pixel, 1 if it is a
    # pixel at the edge
    edgesMask = np.ones_like(skyMask, int)
    edgesMask[int(edges):- int(edges), int(edges):- int(edges)] = 0
    FgDataBackground[(edgesMask==1)] = np.nan

    # removing extreme values. This mask is 0 if it is a good pixel,
    # 1 if it is a pixel with an extreme value
    extremeMask = np.ones_like(skyMask, int)
    fgBackgroundFlux = np.nanmedian(FgDataBackground)
    fgBackgroundSigma = np.nanstd(FgDataBackground)
    extremeMask[np.abs((dataImage-fgBackgroundFlux)/fgBackgroundSigma)<2.99] = 0
    FgDataBackground[(extremeMask==1)] = np.nan

    # Checking values of the background
    fgBackgroundFlux = np.nanmedian(FgDataBackground)
    fgBackgroundSigma = np.nanstd(FgDataBackground)
    fgDataHist, fgDataEdges = np.histogram(FgDataBackground[np.isfinite(FgDataBackground)].flatten(),
                                           bins="fd", density=True)
    # fitting of the histogram
    gaussBest, gaussCovar = curve_fit(manip.gaussian,
                                      fgDataEdges[:-1],
                                      fgDataHist,
                                      p0=[1. / (fgBackgroundSigma * np.sqrt((2. * np.pi))),
                                      fgBackgroundFlux,
                                      fgBackgroundSigma])

    # A second round of global sky subtraction is performed
    print("sourcesFg: A residual background of {:.4f} counts has been removed".format(fgBackgroundFlux))
    fgDataNoBg = np.copy(dataImage) - fgBackgroundFlux

    # Running force photometry on the detected sources
    print("sourcesFg: Aperture photometry on sources with radius {:.4f} pix.".format(radiusNorm))
    fgFluxCent, fgErrFluxCent = manip.quickApPhotmetryNoBg(fgDataNoBg,
                                                           statImage,
                                                           fgXPix,
                                                           fgYPix,
                                                           obj_rad=radiusNorm)

    # Removing sources that are at the edge of the detection at the center
    fgBright = fgFluxCent > .5 * sigSourceDetection * fgErrFluxCent
    fgFluxCent = fgFluxCent[fgBright]
    fgErrFluxCent = fgErrFluxCent[fgBright]
    fgXPix = fgXPix[fgBright]
    fgYPix = fgYPix[fgBright]
    fgAPix = fgAPix[fgBright]
    fgBPix = fgBPix[fgBright]
    fgAngle = fgAngle[fgBright]
    print("sourcesFg: Detected {} sources above {} sigma at the center".format(len(fgXPix), .5*sigSourceDetection))
    # Remove sources at the edge of the FOV
    fgXlocation = (fgXPix > edges) & (fgXPix < (xMax-edges))
    fgYlocation = (fgYPix > edges) & (fgYPix < (yMax-edges))
    fgLocation = fgXlocation & fgYlocation
    fgFluxCent = fgFluxCent[fgLocation]
    fgErrFluxCent = fgErrFluxCent[fgLocation]
    fgXPix = fgXPix[fgLocation]
    fgYPix = fgYPix[fgLocation]
    fgAPix = fgAPix[fgLocation]
    fgBPix = fgBPix[fgLocation]
    fgAngle = fgAngle[fgLocation]
    print("sourcesFg: Detected {} sources after edges removal".format(len(fgXPix)))
    # Removes crazy values axis of elliptical sources
    fgASize = fgAPix < maxSourceSize
    fgBSize = fgBPix < maxSourceSize
    fgAbig   = np.max((fgAPix, fgBPix), axis=0)
    fgAsmall = np.min((fgAPix, fgBPix), axis=0)
    fgEll = 1.-(fgAsmall/fgAbig)
    fgEllCut = (fgEll < np.max([np.percentile(fgEll, 10), maxSourceEll])) & (fgEll > 0.)
    fgShape = fgASize & fgBSize & fgEllCut
    fgFluxCent = fgFluxCent[fgShape]
    fgErrFluxCent = fgErrFluxCent[fgShape]
    fgXPix = fgXPix[fgShape]
    fgYPix = fgYPix[fgShape]
    fgAPix = fgAPix[fgShape]
    fgBPix = fgBPix[fgShape]
    fgAngle = fgAngle[fgShape]
    print("sourcesFg: Removing sources with size larger than {}".format(maxSourceSize))
    print("sourcesFg: Removing sources with ellepticity larger than {}".format(np.max([np.percentile(fgEll, 10), maxSourceEll])))
    print("sourcesFg: Detected {} sources after removing unusal shapes".format(len(fgXPix)))
    del FgDataBackground
    del fgBright
    del fgLocation
    del fgShape

    if debug:
        print("sourcesFg: Saving degub image on {}_fgSourcesMask.pdf".format(output))

        FgDataBackground = np.copy(dataImage)
        skyMask = manip.location(dataImage, fgXPix, fgYPix,
                                 semi_maj=sizeSourceMask * fgAPix,
                                 semi_min=sizeSourceMask * fgBPix,
                                 theta=fgAngle)
        FgDataBackground[(skyMask==1)] = np.nan
        edgesMask = np.ones_like(skyMask, int)
        edgesMask[int(edges):-int(edges), int(edges):-int(edges)] = 0
        FgDataBackground[(edgesMask==1)] = np.nan

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        axImage = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        axMask  = plt.subplot2grid((1, 3), (0, 1), colspan=1)
        axHist  = plt.subplot2grid((1, 3), (0, 2), colspan=1)

        # Plotting field image
        axImage.imshow(dataImage,
                       cmap="Greys", origin="lower",
                       vmin=fgBackgroundFlux - 3. * fgBackgroundSigma,
                       vmax=fgBackgroundFlux + 3. * fgBackgroundSigma)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_title(r"Collapsed image")

        # Plotting field mask
        axMask.imshow(FgDataBackground,
                      cmap="Greys", origin="lower",
                      vmin=fgBackgroundFlux - 3. * fgBackgroundSigma,
                      vmax=fgBackgroundFlux + 3. * fgBackgroundSigma)
        axMask.set_xlabel(r"X [Pixels]", size=30)
        axMask.set_ylabel(r"Y [Pixels]", size=30)
        axMask.set_title(r"Background")

        # Plotting pixel distribution
        axHist.step(fgDataEdges[:-1], fgDataHist, color="gray",
                    zorder=3)
        axHist.plot(fgDataEdges[:-1], manip.gaussian(fgDataEdges[:-1], * gaussBest),
                    color='black', zorder=2)
        axHist.axvline(fgBackgroundFlux, color="black",
                       zorder=1, linestyle=':')
        axHist.set_xlim(left=fgBackgroundFlux - 3. * fgBackgroundSigma,
                        right=fgBackgroundFlux + 3. * fgBackgroundSigma)
        axHist.text(0.52, 0.9, "Med b/g", transform=axHist.transAxes)
        axHist.set_ylabel(r"Pixel Distribution", size=30)
        axHist.set_xlabel(r"Flux", size=30)
        axHist.set_title(r"b/g flux distribution")
        # axHist.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        plt.tight_layout()
        plt.savefig(output+"_fgSourcesMask.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del skyMask
        del edgesMask
        del FgDataBackground

    print("sourcesFg: Filling dictionary with relevant information on the {} sources".format(len(fgXPix)))
    # create dictionary with relevant information on the fg sources
    fgSources = {}
    fgFlux = np.zeros_like(fgXPix)
    fgErrFlux = np.zeros_like(fgXPix)
    fgFluxBg = np.zeros_like(fgXPix)

    # create elliptical mask with all sources masked
    # fgMask is a mask where all sources are masked as 1,
    # while the background is set to 0
    fgMask = manip.location(fgDataNoBg, fgXPix, fgYPix,
                            semi_maj=sizeSourceMask*fgAPix,
                            semi_min=sizeSourceMask*fgBPix,
                            theta=fgAngle)

    for sourceIdx in range(0, len(fgXPix)):
        # create elliptical mask of each source
        # fgThisSourceMask is a mask where the considered source is masked as 1,
        # while the background is set to 0
        fgThisSourceMask = manip.location(fgDataNoBg, fgXPix[sourceIdx], fgYPix[sourceIdx],
                                          semi_maj=sizeSourceMask*fgAPix[sourceIdx],
                                          semi_min=sizeSourceMask*fgBPix[sourceIdx],
                                          theta=fgAngle[sourceIdx])
        fgThisMask, fgThisDataNoBg = np.copy(fgMask), np.copy(fgDataNoBg)
        # fgThisMask is a mask where all sources are masked as 1 BUT the source considered
        fgThisMask[(fgThisSourceMask==1)] = np.int(0)
        # if there is another source in the area of fgThisSourceMask
        # it will be masked within radiusNorm
        for sourceIdx_tmp in range(0, len(fgXPix)):
            if (fgXPix[sourceIdx_tmp] != fgXPix[sourceIdx]):
                fgContaminantSmallMask = manip.location(fgDataNoBg, fgXPix[sourceIdx_tmp], fgYPix[sourceIdx_tmp],
                                                        semi_maj=radiusNorm,
                                                        semi_min=radiusNorm,
                                                        theta=fgAngle[sourceIdx_tmp])
                fgThisMask[(fgContaminantSmallMask==1)] = 1
        # Loading only data from the current source
        fgThisSourceData = np.copy(dataImage)
        fgThisSourceData[(fgThisSourceMask==0)] = 0.
        fgThisSourceDataNoBg = np.copy(fgDataNoBg)
        fgThisSourceDataNoBg[(fgThisSourceMask==0)] = 0.
        fgThisSourceDataNoBg[(fgThisMask==1)] = 0.

        # Loading dictionary with relevant informations
        fgSources[sourceIdx] = {}
        fgSources[sourceIdx]["x"] = fgXPix[sourceIdx]
        fgSources[sourceIdx]["y"] = fgYPix[sourceIdx]
        fgSources[sourceIdx]["a"] = fgAPix[sourceIdx]
        fgSources[sourceIdx]["b"] = fgBPix[sourceIdx]
        fgSources[sourceIdx]["theta"] = fgAngle[sourceIdx]
        fgSources[sourceIdx]["flux"] = fgFluxCent[sourceIdx]
        fgSources[sourceIdx]["errFlux"] = fgErrFluxCent[sourceIdx]
        fgSources[sourceIdx]["radiusFlux"] = radiusNorm
        fgSources[sourceIdx]["fluxBg"] = fgBackgroundFlux
        fgSources[sourceIdx]["errFluxBg"] = fgBackgroundSigma
        fgSources[sourceIdx]["sourceMask"] = fgThisSourceMask
        fgSources[sourceIdx]["contaminantMask"] = fgThisMask
        fgSources[sourceIdx]["sourceData"] = fgThisSourceData
        fgSources[sourceIdx]["sourceDataNoBg"] = fgThisSourceDataNoBg

    """
    # ToDo : this is not working at the moment.
    print("sourcesFg: Saving foreground source dictionary in {}".format(output+"_fgSources.json"))
    with open(output+"_fgSources.json", "w") as jfile:
        fgSourcesJason = muutils.jsonify(fgSources)
        json.dump(fgSourcesJason, jfile, sort_keys=True, indent=4, 
                  separators=(',', ': '), easy_to_read=True, overwrite=True)
    """

    # Deleting temporary images to clear up memory
    del fgThisSourceDataNoBg
    del fgThisSourceData
    del fgThisMask
    del fgThisSourceMask

    return fgSources, fgAllSources, fgBackgroundFlux, fgBackgroundSigma


def cleanFg(datacontainer,
            minChannel=None,
            maxChannel=None,
            maskZ=None,
            maskX=None,
            maskY=None,
            maskXYRad=None,
            sigSourceDetection=5.0,
            minSourceArea=16.,
            sizeSourceMask=6.,
            maxSourceSize=50.,
            maxSourceEll=0.9,
            bgSource=False,
            radiusNorm=1.,
            edges=60,
            output='Object',
            debug=False,
            showDebug=False,
            deepDebug=False):
    """ This macro removes sources from a cube creating a model from the data
    itself. The model will be created collapsing the cube between minChannel
    and maxChannel (considering maskZ as mask for bad channels). The macro
    will search for sources in this image, and create a normalized model for
    each of them.
    Finally, this model will be propagated in the entire cube and then removed.

    Parameters
    ----------
    dataCube : np.array
        data in a 3D array
    statCube : np.array
        variance in a 3D array
    minChannel : int
        min channel to create the image where to detected sources
    maxChannel : int
        max channel to create the image where to detected sources
    maskZ : np.array
        when 1 (or True), this is a channel to be removed
    maskX, maskY, maskXYRad : floats
        x,y location of a source NOT to be subtracted as f/g model.
        Any sources located within maskXYRad from maskX, maskY will
        not be considered in the creation of the model.
    sigSourceDetection : float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    minSourceArea : float
        min area for source detection in the collapsed
        cube. Default is 16.
    sizeSourceMask : float
        for each source, the model will be created in an elliptical
        aperture with size source_mask_size time the semi-minor and semi-major
        axis of the detection. Default is 6.
    maxSourceSize : float
        sources with semi-major or semi-minor axes larger than this
        value will not considered in the foreground source model.
        Default is 50.
    maxSourceEll : float
        sources with ellipticity larger than this value will not
        considered in the foreground source model. Default is 0.9.
    bgSource : bool
        if True, an additional local background subtraction will be
        performed around each detected sources. Default is True
    radiusNorm : float
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
        Dictionary containing relvant informations on the detected
        sources including data, masks, flux, etc.
    """

    print("cleanFg: Starting cleaning of the cube")
    dataCube, statCube = datacontainer.get_data_stat()
    zMax, yMax, xMax = np.shape(dataCube)

    print("cleanFg: Collapsing cube")
    fgData, fgStat = manip.collapse_container(datacontainer, maskZ=maskZ,
                                              min_lambda=minChannel, max_lambda=maxChannel)

    print("cleanFg: Searching for sources in the collapsed cube")
    fgSources, fgAllSources, fgBackgroundFlux, fgBackgroundSigma = sourcesFg(fgData, statImage=fgStat,
                                                                             sigSourceDetection=sigSourceDetection,
                                                                             minSourceArea=minSourceArea,
                                                                             sizeSourceMask=sizeSourceMask,
                                                                             maxSourceSize=maxSourceSize,
                                                                             maxSourceEll=maxSourceEll,
                                                                             radiusNorm=radiusNorm,
                                                                             edges=edges,
                                                                             output=output,
                                                                             debug=debug)

    print("cleanFg: Subtracting contaminants")
    # Subtracting the f/g contamination
    dataCubeClean = np.copy(dataCube)
    dataCubeModel = np.zeros_like(dataCube)
    for sourceIdx in range(0, len(fgSources)):
        print("cleanFg: Removing source {}".format(sourceIdx))
        # creating normalized model
        fgModel = (fgSources[sourceIdx]["sourceDataNoBg"] / fgSources[sourceIdx]["flux"])
        fgModel[(fgSources[sourceIdx]["sourceMask"]==0)] = 0.
        print("         The min, max values for fg model are: {:03.4f}, {:03.4f}".format(np.min(fgModel), np.max(fgModel)))
        # extract spectrum of the source from the dataCube
        fgSourcesExtent = np.max([fgSources[sourceIdx]["a"], fgSources[sourceIdx]["b"]])
        rIbg = 1.1 * sizeSourceMask * (fgSourcesExtent + fgSources[sourceIdx]["radiusFlux"])
        rObg = rIbg + (5. * fgSources[sourceIdx]["radiusFlux"])
        if bgSource:
            fgFluxSource, fgErrFluxSource, fgBgFluxSource = manip.quickSpectrum(dataCube, statcopy=statCube,
                                                                                x_pos=[sourceIdx]["x"],
                                                                                y_pos=[sourceIdx]["y"],
                                                                                radius_pos=fgSources[sourceIdx]["radiusFlux"],
                                                                                inner_rad=rIbg,
                                                                                outer_rad=rObg,
                                                                                void_mask=fgSources[sourceIdx]["contaminantMask"])
        else:
            fgFluxSource, fgErrFluxSource = manip.quickSpectrumNoBg(dataCube, statcopy=statCube,
                                                                    x_pos=fgSources[sourceIdx]["x"],
                                                                    y_pos=fgSources[sourceIdx]["y"],
                                                                    radius_pos=fgSources[sourceIdx]["radiusFlux"])
        keepSource=True
        if maskXYRad is not None:
            distFromMasked = manip.distFromPixel(0., maskY, maskX, 0., np.array(fgSources[sourceIdx]["y"]), np.array(fgSources[sourceIdx]["x"]))
            if distFromMasked < maskXYRad:
                keepSource=False
                print("cleanFg: Source not removed")
                print("         it is located {}<{} pixel away from the XYmask".format(distFromMasked, maskXYRad))

        if keepSource:
            for channel in range(0, zMax):
                # selecting only where the source is significantly detected
                if fgFluxSource[channel] > 0.5 * fgErrFluxSource[channel]:
                    if bgSource:
                        dataCubeClean[channel, :, :] -= ((fgModel * fgFluxSource[channel]) + (fgSources[sourceIdx]["sourceMask"]*fgBgFluxSource[channel]))
                        dataCubeModel[channel, :, :] += ((fgModel * fgFluxSource[channel]) + (fgSources[sourceIdx]["sourceMask"]*fgBgFluxSource[channel]))
                    else:
                        dataCubeClean[channel, :, :] -= (fgModel * fgFluxSource[channel])
                        dataCubeModel[channel, :, :] += (fgModel * fgFluxSource[channel])

        if deepDebug:
            print("cleanFg: Spectrum of the source {}".format(sourceIdx))

            plt.figure(1, figsize=(18, 6))
            gs = gridspec.GridSpec(1, 3)

            axImag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
            axSpec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

            fgModelTempXMin, fgModelTempXMax = np.int(fgSources[sourceIdx]["x"]-rIbg), np.int(fgSources[sourceIdx]["x"]+rIbg)
            fgModelTempYMin, fgModelTempYMax = np.int(fgSources[sourceIdx]["y"]-rIbg), np.int(fgSources[sourceIdx]["y"]+rIbg)
            fgModelTemp = np.copy(fgModel[fgModelTempYMin:fgModelTempYMax, fgModelTempXMin:fgModelTempXMax])
            axImag.imshow(fgModelTemp,
                          cmap="Greys", origin="lower",
                          vmin=0.,
                          vmax=1. / (np.pi * radiusNorm * radiusNorm))
            axImag.set_xlabel(r"X [Pixels]", size=30)
            axImag.set_ylabel(r"Y [Pixels]", size=30)

            axSpec.plot(fgFluxSource, color='black', zorder=3, label='Flux')
            axSpec.plot(fgErrFluxSource, color='gray', alpha=0.5, zorder=2, label='Error')
            if bgSource:
                axSpec.plot(fgBgFluxSource * (np.pi * radiusNorm * radiusNorm), color='red', alpha=0.5, zorder=1, label='b/g')
            axSpec.legend()
            axSpec.set_xlabel(r"Channel", size=30)
            axSpec.set_ylabel(r"Flux", size=30)

            plt.tight_layout()
            plt.show()
            plt.close()
            del fgModelTemp

    print("cleanFg: Source cleaning performed")

    if debug:
        print("cleanFg: Saving degub image on {}_fgSourcesCleaned.pdf".format(output))
        fgDataClean = manip.collapse_cube(dataCubeClean, maskZ=maskZ,
                                          min_lambda=minChannel, max_lambda=maxChannel)
        fgStatClean = manip.collapse_cube(statCube, maskZ=maskZ,
                                          min_lambda=minChannel, max_lambda=maxChannel)
        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axClean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        axImage.imshow(fgData,
                       cmap="Greys", origin="lower",
                       vmin=fgBackgroundFlux-3.*fgBackgroundSigma,
                       vmax=fgBackgroundFlux+3.*fgBackgroundSigma)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_title(r"Collapsed image")

        for sourceIdx in range(0,len(fgSources)):
            fgSourcesArtist = Ellipse(xy=(fgSources[sourceIdx]["x"], fgSources[sourceIdx]["y"]),
                                      width=sizeSourceMask*fgSources[sourceIdx]["a"],
                                      height=sizeSourceMask*fgSources[sourceIdx]["b"],
                                      angle=fgSources[sourceIdx]["theta"])
            fgSourcesArtist.set_facecolor("none")
            fgSourcesArtist.set_edgecolor("red")
            fgSourcesArtist.set_alpha(0.8)
            axImage.add_artist(fgSourcesArtist)

        # Plotting cleaned image
        axClean.imshow(fgDataClean,
                       cmap="Greys", origin="lower",
                       vmin=fgBackgroundFlux-3.*fgBackgroundSigma,
                       vmax=fgBackgroundFlux+3.*fgBackgroundSigma)
        axClean.set_xlabel(r"X [Pixels]", size=30)
        axClean.set_ylabel(r"Y [Pixels]", size=30)
        axClean.set_title(r"Cleaned image")

        plt.tight_layout()
        plt.savefig(output+"_fgSourcesCleaned.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()

    print("cleanFg: Saving source list on {}_fgSources.txt".format(output))
    f = open(output+"_fgSources.txt", 'w')
    f.write("Idx fgXPix fgYPix fgAPix fgBPix fgTheta\n")
    for sourceIdx in range(0,len(fgSources)):
        f.write("{:.0f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(sourceIdx, fgSources[sourceIdx]["x"], fgSources[sourceIdx]["y"], fgSources[sourceIdx]["a"], fgSources[sourceIdx]["b"], fgSources[sourceIdx]["theta"]))
    f.close()

    gc.collect()
    return dataCubeClean, dataCubeModel, fgSources


def makePsf(datacontainer,
            x_pos,
            y_pos,
            min_lambda=None,
            max_lambda=None,
            maskZ=None,
            radius_pos=2.,
            inner_rad=10.,
            outer_rad=15.,
            rPsf=50.,
            cType="sum",
            norm=True,
            debug=False,
            showDebug=False):
    """Given a Cube, the macro collapses it along the z-axis between minChannel and
    maxChannel. If maskZ is given, channels masked as 1 (or True) are removed.
    if cType is set to 'average', the macro uses to STAT information to perform a
    weighted mean along the velocity axis. In other words, each spaxel of the resulting
    image will be the weighted mean spectrum of that spaxels along the wavelengths.
    If norm is 'True' the macro normalize the flux of the PSF within radius_pos = 1.

    Parameters
    ----------
    datacontainer : IFUcube Object
        data read in from cubeClass.py
    x_pos : float
        x-location of the source in pixel
    y_pos : float
        y-location of the source in pixel
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
    rPsf : float
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
    if cType == 'sum':
        print("makePsf: Summing channels")
        psf_data, psf_stat = manip.collapse_container(datacontainer, min_lambda, max_lambda)
    else:
        print("makePsf: Average combining channels")
        psf_data, psf_stat = manip.collapse_mean_container(datacontainer, min_lambda, max_lambda)

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
                              semi_maj=rPsf, semi_min=rPsf)
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
        ax_image.set_xlim(left=x_pos - rPsf, right=x_pos + rPsf)
        ax_image.set_ylim(bottom=y_pos - rPsf, top=y_pos + rPsf)
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
        '''
                         vmin=np.nanmin(psf_stat),
                         vmax=(0.5 * np.nanmax(psf_stat)))
        '''
        ax_stat.set_xlabel(r"X [Pixels]", size=30)
        ax_stat.set_ylabel(r"Y [Pixels]", size=30)
        ax_stat.set_xlim(left=x_pos - rPsf, right=x_pos + rPsf)
        ax_stat.set_ylim(bottom=y_pos - rPsf, top=y_pos + rPsf)
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
    one within rPsf and that the PSF model is centered in the same location
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

    print("cleanPsf: PSF subtraction on cube")
    datacopy, statcopy = datacontainer.get_data_stat()
    datacopy_model = np.zeros_like(datacopy)
    z_max, y_max, x_max = np.shape(datacopy)

    print("cleanPsf: The min, max values for PSF model are: {:03.4f}, {:03.4f}".format(np.min(psf_model),
                                                                                       np.max(psf_model)))
    # extract spectrum of the source from the dataCube
    if bg_psf:
        flux_source, err_flux_source, bg_flux_source = manip.quickSpectrum(datacontainer=datacopy,
                                                                           statcopy=statcopy,
                                                                           x_pos=x_pos, y_pos=y_pos,
                                                                           radius_pos=radius_pos,
                                                                           inner_rad=inner_rad,
                                                                           outer_rad=outer_rad)
    else:
        flux_source, err_flux_source = manip.quickSpectrumNoBg(datacontainer=datacopy,
                                                               statcopy=statcopy,
                                                               x_pos=x_pos, y_pos=y_pos,
                                                               radius_pos=radius_pos)
        bg_flux_source = np.zeros_like(flux_source)

    for channel in range(0, z_max):
        # selecting only where the source is significantly detected
        if flux_source[channel] > 0.5 * (err_flux_source[channel]):
            datacopy[channel, :, :] -= ((psf_model * flux_source[channel]) + bg_flux_source[channel])
            datacopy_model[channel, :, :] += ((psf_model * flux_source[channel]) + bg_flux_source[channel])

    if debug:
        print("cleanPsf: Spectrum of the source")

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
            ax_spec.plot(bg_flux_source * (np.pi * radius_pos * radius_pos), color='red', alpha=0.5, zorder=1, label='b/g')
        ax_spec.legend()
        ax_spec.set_xlabel(r"Channel", size=30)
        ax_spec.set_ylabel(r"Flux", size=30)

        plt.tight_layout()
        if show_debug:
            plt.show()
        plt.close()
        del model_temp

    print("cleanPsf: PSF cleaning performed")

    gc.collect()
    return datacopy, datacopy_model
