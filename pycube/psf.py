import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import gridspec

import numpy as np

from pycube.core import manip
from pycube.core import background

import sep
import gc
from IPython import embed


def find_sources(datacube, statcube=None,
                 min_lambda=None, max_lambda=None,
                 var_factor=5.,
                 sig_detect=3.501,
                 min_area=16.,
                 gain=1.1,  # get it from the header
                 deblend_val=0.005):
    """
    Automated scanning of given data and identifies good sources.
    If data is in 3D format, function will collapse given wavelength parameters

    Inputs:
        datacube (array):
            data cube. 2D or 3D
        statcube (array):
            variance cube. 2D or 3D. optional ~ will generate from data if not passed
            defaults to None
        min_lambda (float):
            minimum wavelength value to collapse 3D image, default is None
        max_lambda (float):
            maximum wavelength value to collapse 3D image, default is None
        var_factor (int / float):
            affects generated variance, if variance is auto-generated from image data, default 5.
        sig_detect (int / float):
            minimum signal detected by function, default 3.5
        min_area (int / float):
            minimum area determined to be a source, default 16
        gain:
            can be pulled from Fits file, default 1.1
        deblend_val:
            value for sep extractor, minimum contrast ratio for object blending, default 0.005
    Returns:
        xPix (np.array):
        yPix (np.array):
        aPix (np.array):
        bPix (np.array):
        angle (np.array):
        all_objects (np.array):
    """
    data_background = np.copy(datacube)
    data_background = manip.check_collapse(data_background, min_lambda, max_lambda)

    if statcube is not None:
        var_background = np.copy(statcube)
        var_background = manip.check_collapse(var_background, min_lambda, max_lambda)
    else:
        # implemented for data without variance (stat) file
        print("No variance detected. Generating from data..")
        variance_shell = np.zeros_like(data_background)
        med_background = np.nanmedian(data_background)
        std_image = np.nanstd(data_background - med_background)
        var_background = variance_shell + \
                         np.nanvar(data_background[(data_background - med_background) < (var_factor * std_image)])

    image_background = background.sextractor_background(data_background, var_background, var_value=7.)
    void_background = data_background - image_background

    print("find_sources: Searching sources {}-sigma above noise".format(sig_detect))
    all_objects = sep.extract(void_background, sig_detect,
                              var=var_background,
                              minarea=min_area,
                              filter_type='matched',
                              gain=gain,
                              clean=True,
                              deblend_cont=deblend_val,
                              filter_kernel=None)
    # Sorting sources by flux at the peak
    index_by_flux = np.argsort(all_objects['peak'])[::-1]
    all_objects = all_objects[index_by_flux]
    good_sources = all_objects['flag'] < 1
    x_pos = np.array(all_objects['x'][good_sources])
    y_pos = np.array(all_objects['y'][good_sources])
    maj_axis = np.array(all_objects['a'][good_sources])
    min_axis = np.array(all_objects['b'][good_sources])
    angle = np.array(all_objects['theta'][good_sources])
    print("find_sources: {} good sources detected".format(np.size(x_pos)))
    # preserve memory
    del image_background
    del void_background
    del good_sources
    return x_pos, y_pos, maj_axis, min_axis, angle, all_objects


def statBg(dataCube,
           statCube=None,
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
    """
    This estimates the sky background of a MUSE cube after removing sources.
    Sources are detected in an image created by collapsing the cube between minChannel
    and maxChannel (considering maskZ as mask for bad channels).
    Average, std, and median will be saved.

    Inputs:
        dataCube (np.array):
            data in a 3D array
        statCube (np.array):
            variance in a 3D array
        min_lambda (int):
            min channel to create the image where to detect sources
        max_lambda (int):
            max channel to create the image where to detect sources
        maskZ
            when 1 (or True), this is a channel to be removed
        maskXY
            when 1 (or True), this spatial pixel will remove from
            the estimate of the b/g values
        sigSourceDetection (float):
            detection sigma threshold for sources in the
            collapsed cube. Defaults is 5.0
        minSourceArea (float):
            min area for source detection in the collapsed
            cube. Default is 16.
        sizeSourceMask (float):
            for each source, the model will be created in an elliptical
            aperture with size sizeSourceMask time the semi-minor and semi-major
            axis of the detection. Default is 6.
        maxSourceSize (float):
            sources with semi-major or semi-minor axes larger than this
            value will not be considered in the foreground source model.
            Default is 50.
        maxSourceEll (float):
            sources with ellipticity larger than this value will not be
            considered in the foreground source model. Default is 0.9.
        edges (int):
            frame size removed to avoid problems related to the edge
            of the image
        output (string):
            root file name for outputs
    Returns:
        averageBg, medianBg, stdBg, varBg, pixelsBg (np.array):
            average, median, standard deviation, variance, and number of pixels after masking
            sources, edges, and NaNs of the background.
        maskBg2D (np.array):
            2D mask used to determine the background region. This mask has 1 if there is
            a source or is on the edge of the cube. It is 0 if the pixel is considered
            in the background estimate.
    """

    print("statBg: Starting estimate of b/g stats")

    print("statBg: Collapsing cube")
    datacopy = np.copy(dataCube)
    statcopy = np.copy(statCube)
    data_image = manip.collapse_cube(datacopy, min_lambda, max_lambda)
    stat_image = manip.collapse_cube(statcopy, min_lambda, max_lambda)
    print("statBg: Searching for sources in the collapsed cube")
    x_pos, y_pos, maj_axis, min_axis, angle, all_objects = find_sources(data_image, stat_image, sig_detect=sigSourceDetection,
                                                             min_area=minSourceArea)

    print("statBg: Detected {} sources".format(len(x_pos)))
    print("statBg: Masking sources")
    maskBg2D = np.zeros_like(data_image)

    sky_mask = manip.location(data_image, x_position=x_pos, y_position=y_pos,
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
    mask_Bg_3D = np.broadcast_to((maskBg2D == 1), datacopy.shape)
    datacopy[(mask_Bg_3D == 1)] = np.nan
    averageBg, stdBg, medianBg, varBg, pixelsBg = np.nanmean(datacopy, axis=(1, 2)), \
                                                  np.nanstd(datacopy, axis=(1, 2)), \
                                                  np.nanmedian(datacopy, axis=(1, 2)), \
                                                  np.nanvar(datacopy, axis=(1, 2)), \
                                                  np.count_nonzero(~np.isnan(datacopy), axis=(1, 2))
    bgDataImage = np.copy(data_image)
    bgDataImage[(maskBg2D == 1)] = np.nan

    if debug:
        print("statBg: Saving debug image on {}_BgRegion.pdf".format(output))
        bgStatImage = manip.collapse_cube(statcopy, min_lambda, max_lambda)
        tempBgFlux = np.nanmean(bgDataImage)
        tempBgStd = np.nanstd(bgDataImage)

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axClean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        axImage.imshow(data_image,
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
        del bgStatImage

    # preserve memory
    del data_image
    del datacopy
    del mask_Bg_3D
    del sky_mask
    del edges_mask
    del stat_image
    del statcopy
    del all_objects

    gc.collect()

    return averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D, bgDataImage


def subtractBg(datacube,
               statcube=None,
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
    """
    This macro remove residual background in the cubes and fix the variance
    vector after masking sources. Sources are detected in an image created by
    collapsing the cube between min_lambda and max_lambda (considering maskZ as
    mask for bad channels). If statCube is none, it will be created and for each
    channel, the variance of the background will be used.

   Inputs:
        datacube (np.array):
            data in a 3D array
        statcube (np.array):
            variance in a 3D array
        min_lambda (int):
            min channel to create the image to detected sources
        max_lambda (int):
            max channel to create the image to detected sources
        maskZ
            when 1 (or True), this is a channel to be removed while
            collapsing the cube to detect sources
        maskXY
            when 1 (or True), this spatial pixel will be removed from
            the estimate of the b/g values
        sigSourceDetection (float):
            detection sigma threshold for sources in the
            collapsed cube. Defaults is 5.0
        minSourceArea (float):
            min area for source detection in the collapsed
            cube. Default is 16.
        sizeSourceMask (float):
            for each source, the model will be created in an elliptical
            aperture with size sizeSourceMask time the semi-minor and semi-major
            axis of the detection. Default is 6.
        maxSourceSize (float):
            sources with semi-major or semi-minor axes larger than this
            value will not be considered in the foreground source model.
            Default is 50.
        maxSourceEll (float):
            sources with ellipticity larger than this value will not
            be considered in the foreground source model. Default is 0.9.
        edges (int):
            frame size removed to avoid problems related to the edge
            of the image
        output (string):
            root file name for outputs
    Returns:
        dataCubeBg, statCubeBg (np.array):
            3D data and variance cubes after residual sky subtraction and
            variance rescaled to match the background variance.
        averageBg, medianBg, stdBg, varBg, pixelsBg (np.array):
            average, median, standard deviation, variance, and number of pixels after masking
            sources, edges, and NaNs of the background.
        maskBg2D (np.array):
            2D mask used to determine the background region. This mask has 1 if there is
            a source or is on the edge of the cube. It is 0 if the pixel is considered
            in the background estimate.
    """

    print("subtractBg: Starting the procedure to subtract the background")

    # Getting spectrum of the background
    averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D, bgDataImage = statBg(datacube,
                                                                                statCube=statcube,
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

    print("subtractBg: Subtracting background from dataCube")
    datacopy = np.copy(datacube)
    z_max, y_max, x_max = np.shape(datacube)
    for channel in range(0, z_max):
        datacopy[channel, :, :] -= medianBg[channel]

    if statcube is None:
        print("subtractBg: Creating statCube with variance inferred from background")
        statcopy = np.copy(datacube)
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


def makePsf(datacube,
            statcube,
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
    """
    Given a Cube, the macro collapses it along the z-axis between minChannel and
    maxChannel. If maskZ is given, channels masked as 1 (or True) are removed.
    if cType is set to 'average', the macro uses to STAT information to perform a
    weighted mean along the velocity axis. In other words, each spaxel of the resulting
    image will be the weighted mean spectrum of that spaxels along the wavelengths.
    If norm is 'True' the macro normalize the flux of the PSF within radius_pos = 1.

   Inputs:
        datacube (np.array):
            data in a 3D array
        statcube (np.array):
            variance in a 3D array
        xObj (float):
            x-location of the source in pixel
        yObj (float):
            y-location of the source in pixel
        min_lambda (int):
            min channel to create collapsed image
        max_lambda (int):
            max channel to create collapsed image
        maskZ
            when 1 (or True), this is a channel to be removed
        rObj (float):
            radius where to perform the aperture photometry
        rIbg (float):
            inner radius of the background region in pixel
        rObg (float):
            outer radius of the background region in pixel
        rPsf (float):
            radius of the PSF image to be created. Outside
            this pixels values are set to zero
        cType(str):
            type of combination for PSF creation:
            'average' is weighted average
            'sum' is direct sum of all pixels
        norm (bool):
            if 'True' normalizes the central regions of the
            PSF to 1.
    Returns:
        psfData, psfStat (np.array):
            PSF data and variance images
    """

    print("makePsf: Creating PSF model")

    if cType == 'sum':
        print("makePsf: Summing channels")
        psf_data = manip.collapse_cube(datacube, min_lambda=min_lambda, max_lambda=max_lambda)
        psf_stat = manip.collapse_cube(statcube, min_lambda=min_lambda, max_lambda=max_lambda)
    else:
        print("makePsf: Average combining channels")
        psf_data, psf_stat = manip.collapse_mean_cube(datacube, statcube,
                                                      min_lambda=min_lambda, max_lambda=max_lambda)

    psf_flux, psf_err_flux, psf_ave_flux = manip.quickApPhotmetry(psf_data, psf_stat, x_pos=x_pos, y_pos=y_pos,
                                                                  radius_pos=radius_pos, inner_rad=inner_rad, outer_rad=outer_rad)

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
        #gs = gridspec.GridSpec(1, 2)

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
        plt.show()
        plt.close()

    gc.collect()
    return psf_data, psf_stat
