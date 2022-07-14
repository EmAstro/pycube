import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import gridspec

import numpy as np

from pycube.core import manip
from pycube.core import background
from pycube.instruments import instrument

import sep
import gc
from IPython import embed


def find_sources(datacontainer,
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

    data_background, var_background = manip.collapse_container(datacontainer, min_lambda=min_lambda,
                                                          max_lambda=max_lambda, var_thresh=var_factor)

    image_background = background.sextractor_background(data_background, var_background, threshold)
    void_background = data_background - image_background

    # print("find_sources: Searching sources {}-sigma above noise".format(sig_detect))
    all_objects = sep.extract(void_background, sig_detect,
                              var=var_background,
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
                    sigSourceDetection=5.0,
                    minSourceArea=16.,
                    sizeSourceMask=6.,
                    maxSourceSize=50.,
                    maxSourceEll=0.9,
                    edges=10):
    """

    Parameters
    ----------
    datacontainer:

    sigSourceDetection:

    minSourceArea:
    sizeSourceMask:
    maxSourceSize:
    maxSourceEll:
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
                                                                            sig_detect=sigSourceDetection,
                                                                            min_area=minSourceArea)
        maskBg2D = np.zeros_like(tmpdatacopy)
        sky_mask = manip.location(tmpdatacopy,
                                  x_position=x_pos, y_position=y_pos,
                                  semi_maj=sizeSourceMask * maj_axis,
                                  semi_min=sizeSourceMask * min_axis,
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
        aperture with size sizeSourceMask time the semi-minor and semi-major
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
        aperture with size sizeSourceMask time the semi-minor and semi-major
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
    dataCubeBg, statCubeBg : np.array
        3D data and variance cubes after residual sky subtraction and
        variance rescaled to match the background variance.
    averageBg, medianBg, stdBg, varBg, pixelsBg : np.array
        average, median, standard deviation, variance, and number of pixels after masking
        sources, edges, and NaNs of the background.
    maskBg2D : np.array
        2D mask used to determine the background region. This mask has 1 if there is
        a source or is on the edge of the cube. It is 0 if the pixel is considered
        in the background estimate.
    """

    print("subtractBg: Starting the procedure to subtract the background")

    # Getting spectrum of the background
    averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D, bgDataImage = statBg(datacontainer,
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
        datacopy[channel, :, :] -= medianBg[channel]
    if statcube is None:
        print("subtractBg: Creating statCube with variance inferred from background")
        statcopy = np.copy(datacontainer)
        for channel in range(0, z_max):
            statcopy[channel, :, :] = varBg[channel]
    else:
        print("subtractBg: Estimating correction for statCube variance")
        # Removing sources and edges
        statcopy_Nan = np.copy(statcube)
        maskBg3D = np.broadcast_to((maskBg2D == 1), statcopy_Nan.shape)
        statcopy_Nan[(maskBg3D == 1)] = np.nan
        # Calculating average variance per channel
        averageStatBg = np.nanmean(statcopy_Nan, axis=(1, 2))
        del statcopy_Nan
        del maskBg3D
        # Rescaling cube variance
        scaleFactor = varBg / averageStatBg
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

        print("subtractBg: The average value subtracted to the b/g level is {:.5f}".format(np.average(medianBg)))
        if debug:
            manip.nicePlot()
            plt.figure(1, figsize=(9, 6))
            plt.plot(range(0, z_max), medianBg, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Median Background")
            plt.axhline(np.average(medianBg))
            plt.savefig(output + "_bgCorrection.pdf", dpi=400.,
                        format="pdf", bbox_inches="tight")
            if showDebug:
                plt.show()
            plt.close()

    gc.collect()
    return datacopy, statcopy, averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D, bgDataImage


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
        ax_stat.imshow(psf_stat,
                       cmap="Greys", origin="lower",
                       vmin=np.nanmin(psf_stat),
                       vmax=0.5 * np.nanmax(psf_stat))
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


def cleanPsf(datacontainer,
             psfModel,
             x_pos,
             y_pos,
             radius_pos=2.,
             inner_rad=10.,
             outer_rad=15.,
             bgPsf=True,
             debug=False,
             showDebug=False):
    """Given a cube and a PSF model, the macro subtracts the PSF contribution
    along the wavelength axis. It assumes that the PSF model is normalized to
    one within rPsf and that the PSF model is centered in the same location
    of the object you want to remove. This will be improved in the future.

    Parameters
    ----------
    datacontainer : IFUcube object
        Data initialized in cubeClass.py
    psfModel : np.array
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
    bgPsf : bool
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

    print("cleanPsf: The min, max values for PSF model are: {:03.4f}, {:03.4f}".format(np.min(psfModel),
                                                                                       np.max(psfModel)))
    # extract spectrum of the source from the dataCube
    if bgPsf:
        fluxSource, errFluxSource, bgFluxSource = manip.quickSpectrum(datacontainer=datacopy,
                                                                      statcopy=statcopy,
                                                                      x_pos=x_pos, y_pos=y_pos,
                                                                      radius_pos=radius_pos,
                                                                      inner_rad=inner_rad,
                                                                      outer_rad=outer_rad)
    else:
        fluxSource, errFluxSource = manip.quickSpectrumNoBg(datacontainer=datacopy,
                                                            statcopy=statcopy,
                                                            x_pos=x_pos,y_pos=y_pos,
                                                            radius_pos=radius_pos)

    for channel in range(0, z_max):
        # selecting only where the source is significantly detected
        if fluxSource[channel] > 0.5 * (errFluxSource[channel]):
            if bgPsf:
                datacopy[channel, :, :] -= ((psfModel * fluxSource[channel]) + bgFluxSource[channel])
                datacopy_model[channel, :, :] += ((psfModel * fluxSource[channel]) + bgFluxSource[channel])
            else:
                datacopy[channel, :, :] -= (psfModel * fluxSource[channel])
                datacopy_model[channel, :, :] += (psfModel * fluxSource[channel])

    if debug:
        print("cleanPsf: Spectrum of the source")

        manip.nicePlot()

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        axImag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        axSpec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

        modelTempXMin, modelTempXMax = int(x_pos - inner_rad), int(x_pos + inner_rad)
        modelTempYMin, modelTempYMax = int(y_pos - inner_rad), int(y_pos + inner_rad)
        modelTemp = np.copy(psfModel[modelTempYMin:modelTempYMax, modelTempXMin:modelTempXMax])
        axImag.imshow(modelTemp,
                      cmap="Greys", origin="lower",
                      vmin=0.,
                      vmax=1. / (np.pi * radius_pos * radius_pos))
        axImag.set_xlabel(r"X [Pixels]", size=30)
        axImag.set_ylabel(r"Y [Pixels]", size=30)

        axSpec.plot(fluxSource, color='black', zorder=3, label='Flux')
        axSpec.plot(errFluxSource, color='gray', alpha=0.5, zorder=2, label='Error')
        if bgPsf:
            axSpec.plot(bgFluxSource * (np.pi * radius_pos * radius_pos), color='red', alpha=0.5, zorder=1, label='b/g')
        axSpec.legend()
        axSpec.set_xlabel(r"Channel", size=30)
        axSpec.set_ylabel(r"Flux", size=30)

        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()
        del modelTemp

    print("cleanPsf: PSF cleaning performed")

    gc.collect()
    return datacopy, datacopy_model
