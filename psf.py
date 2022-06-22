import numpy as np
from pycube.core import manip
from pycube.core import background
import sep


# working on setting this to be user defined.
def findSources(datacube, statcube,
                sigDetect=3.501,
                minArea=16.):
    print("findSources: Starting sources detection")
    print("findSources: Creating background model")
    data_background = np.copy(datacube)
    image_background = background.sextractor_background(datacube, statcube)

    void_background = data_background - image_background
    print("findSources: Searching sources {}-sigma above noise".format(sigDetect))
    all_objects = sep.extract(void_background, sigDetect,
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
    del image_background
    del void_background
    del good_sources
    return xPix, yPix, aPix, bPix, angle, all_objects

def statBg(dataCube,
           statCube=None,
           minChannel=None,
           maxChannel=None,
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
    """ This estimates the sky background of a MUSE cube after removing sources.
    Sources are detected in a image created by collapsing the cube between minChannel
    and maxChannel (considering maskZ as mask for bad channels).
    Average, std, and median will be saved.
    Parameters
    ----------
    dataCube : np.array
        data in a 3D array
    statCube : np.array
        variance in a 3D array
    minChannel : np.int
        min channel to create the image where to detected sources
    maxChannel : np.int
        max channel to create the image where to detected sources
    maskZ
        when 1 (or True), this is a channel to be removed
    maskXY
        when 1 (or True), this spatial pixel will removed from
        the estimate of the b/g values
    sigSourceDetection : np.float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    minSourceArea : np.float
        min area for source detection in the collapsed
        cube. Default is 16.
    sizeSourceMask : np.float
        for each source, the model will be created in an elliptical
        aperture with size sizeSourceMask time the semi-minor and semi-major
        axis of the detection. Default is 6.
    maxSourceSize : np.float
        sources with semi-major or semi-minor axes larger than this
        value will not considered in the foreground source model.
        Default is 50.
    maxSourceEll : np.float
        sources with ellipticity larger than this value will not
        considered in the foreground source model. Default is 0.9.
    edges : np.int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for outputs
    Returns
    -------
    averageBg, medianBg, stdBg, varBg, pixelsBg : np.arrays
        average, median, standard deviation, variance, and number of pixels after masking
        sources, edges, and NaNs of the background.
    maskBg2D : np.array
        2D mask used to determine the background region. This mask has 1 if there is
        a source or is on the edge of the cube. It is 0 if the pixel is considered
        in the background estimate.
    """

    print("statBg: Starting estimate of b/g stats")

    print("statBg: Collapsing cube")
    dataImage = manip.collapse_cube(dataCube, minChannel, maxChannel)
    statImage = manip.collapse_cube(statCube, minChannel, maxChannel)
    print("statBg: Searching for sources in the collapsed cube")
    xPix, yPix, aPix, bPix, angle, allSources = manip.location(dataImage, statImage,
                                                                    sigDetect=sigSourceDetection,
                                                                    minArea=minSourceArea)

    print("statBg: Detected {} sources".format(len(xPix)))

    print("statBg: Masking sources")
    maskBg2D = np.zeros_like(dataImage)
    skyMask = muutils.ellipticalMask(dataImage, xPix, yPix,
                                     aObj=sizeSourceMask*aPix,
                                     bObj=sizeSourceMask*bPix,
                                     thetaObj=angle)
    maskBg2D[(skyMask==1)] = np.int(1)

    print("statBg: Masking Edges")
    # removing edges. This mask is 0 if it is a good pixel, 1 if it is a
    # pixel at the edge
    edgesMask = np.ones_like(skyMask, dtype=np.int_)
    edgesMask[np.int(edges):-np.int(edges),np.int(edges):-np.int(edges)] = np.int(0)
    maskBg2D[(edgesMask==1)] = np.int(1)

    if maskXY is not None:
        print("statBg: Masking spatial pixels from input maskXY")
        maskBg2D[(maskXY==1)] = np.int(1)

    print("statBg: Performing b/g statistic")
    bgCube = np.copy(dataCube)
    maskBg3D = np.broadcast_to((maskBg2D==1), bgCube.shape)
    bgCube[(maskBg3D==1)] = np.nan
    averageBg, stdBg, medianBg, varBg, pixelsBg = np.nanmean(bgCube, axis=(1,2)), np.nanstd(bgCube, axis=(1,2)), np.nanmedian(bgCube, axis=(1,2)), np.nanvar(bgCube, axis=(1,2)), np.count_nonzero(~np.isnan(bgCube), axis=(1,2))

    if debug:
        print("statBg: Saving degub image on {}_BgRegion.pdf".format(output))
        bgDataImage, bgStatImage = muutils.collapseCube(bgCube, statCube=statCube, maskZ=maskZ,
                                                        minChannel=minChannel, maxChannel=maxChannel)
        tempBgFlux = np.nanmean(bgDataImage)
        tempBgStd = np.nanstd(bgDataImage)

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axClean = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        axImage.imshow(dataImage,
                       cmap="Greys", origin="lower",
                       vmin=tempBgFlux-3.*tempBgStd,
                       vmax=tempBgFlux+3.*tempBgStd)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_title(r"Collapsed image")

        # Plotting background image
        axClean.imshow(bgDataImage,
                       cmap="Greys", origin="lower",
                       vmin=tempBgFlux-3.*tempBgStd,
                       vmax=tempBgFlux+3.*tempBgStd)
        axClean.set_xlabel(r"X [Pixels]", size=30)
        axClean.set_ylabel(r"Y [Pixels]", size=30)
        axClean.set_title(r"b/g image")

        plt.tight_layout()
        plt.savefig(output+"_BgRegion.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del bgDataImage
        del bgStatImage

    del dataImage
    del bgCube
    del maskBg3D
    del skyMask
    del edgesMask
    del statImage
    del allSources

    gc.collect()
    return averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D

def subtractBg(dataCube,
               statCube=None,
               minChannel=None,
               maxChannel=None,
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
    """ This macro remove residual background in the cubes and fix the variance
    vector after masking sources. Sources are detected in a image created by
    collapsing the cube between minChannel and maxChannel (considering maskZ as
    mask for bad channels). If statCube is none, it will be created and for each
    channel, the variance of the background will be used.
    Parameters
    ----------
    dataCube : np.array
        data in a 3D array
    statCube : np.array
        variance in a 3D array
    minChannel : np.int
        min channel to create the image where to detected sources
    maxChannel : np.int
        max channel to create the image where to detected sources
    maskZ
        when 1 (or True), this is a channel to be removed while
        collapsing the cube to detect sources
    maskXY
        when 1 (or True), this spatial pixel will removed from
        the estimate of the b/g values
    sigSourceDetection : np.float
        detection sigma threshold for sources in the
        collapsed cube. Defaults is 5.0
    minSourceArea : np.float
        min area for source detection in the collapsed
        cube. Default is 16.
    sizeSourceMask : np.float
        for each source, the model will be created in an elliptical
        aperture with size sizeSourceMask time the semi-minor and semi-major
        axis of the detection. Default is 6.
    maxSourceSize : np.float
        sources with semi-major or semi-minor axes larger than this
        value will not considered in the foreground source model.
        Default is 50.
    maxSourceEll : np.float
        sources with ellipticity larger than this value will not
        considered in the foreground source model. Default is 0.9.
    edges : np.int
        frame size removed to avoid problems related to the edge
        of the image
    output : string
        root file name for outputs
    Returns
    -------
    dataCubeBg, statCubeBg : np.arrays
        3D data and variance cubes after residual sky subtraction and
        variance rescaled to match the background variance.
    averageBg, medianBg, stdBg, varBg, pixelsBg : np.arrays
        average, median, standard deviation, variance, and number of pixels after masking
        sources, edges, and NaNs of the background.
    maskBg2D : np.array
        2D mask used to determine the background region. This mask has 1 if there is
        a source or is on the edge of the cube. It is 0 if the pixel is considered
        in the background estimate.
    """

    print("subtractBg: Starting the procedure to subtract the background")

    # Getting spectrum of the background
    averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D = statBg(dataCube,
                                                                   statCube=statCube,
                                                                   minChannel=minChannel,
                                                                   maxChannel=maxChannel,
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
    dataCubeBg = np.copy(dataCube)
    zMax, yMax, xMax = np.shape(dataCube)
    for channel in range(0,zMax):
            dataCubeBg[channel,:,:] = dataCube[channel,:,:] - medianBg[channel]

    if statCube is None:
        print("subtractBg: Creating statCube with variance inferred from background")
        statCubeBg = np.copy(dataCube)
        zMax, yMax, xMax = np.shape(dataCube)
        for channel in range(0,zMax):
            statCubeBg[channel,:,:] = varBg[channel]
    else:
        print("subtractBg: Estimating correction for statCube variance")
        # Removing sources and edges
        statCubeBgNan = np.copy(statCube)
        maskBg3D = np.broadcast_to((maskBg2D==1), statCubeBgNan.shape)
        statCubeBgNan[(maskBg3D==1)] = np.nan
        # Calculating average variance per channel
        averageStatBg = np.nanmean(statCubeBgNan, axis=(1,2))
        del statCubeBgNan
        del maskBg3D
        # Rescaling cube variance
        scaleFactor = varBg / averageStatBg
        statCubeBg = np.copy(statCube)
        zMax, yMax, xMax = np.shape(statCubeBg)
        for channel in range(0,zMax):
            statCubeBg[channel,:,:] = statCube[channel,:,:] * scaleFactor[channel]
        print("subtractBg: The average correction factor for variance is {:.5f}".format(np.average(scaleFactor)))
        if debug:
            muutils.nicePlot()
            plt.figure(1, figsize=(9,6))
            plt.plot(range(0,zMax), scaleFactor, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Correction Factor")
            plt.axhline(np.average(scaleFactor))
            plt.savefig(output+"_VarianceCorrection.pdf", dpi=400.,
                        format="pdf", bbox_inches="tight")
            if showDebug:
                plt.show()
            plt.close()

        print("subtractBg: The average value subtracted to the b/g level is {:.5f}".format(np.average(medianBg)))
        if debug:
            muutils.nicePlot()
            plt.figure(1, figsize=(9,6))
            plt.plot(range(0,zMax), medianBg, color='black')
            plt.xlabel(r"Channels")
            plt.ylabel(r"Median Background")
            plt.axhline(np.average(medianBg))
            plt.savefig(output+"_bgCorrection.pdf", dpi=400.,
                        format="pdf", bbox_inches="tight")
            if showDebug:
                plt.show()
            plt.close()


    gc.collect()
    return dataCubeBg, statCubeBg, averageBg, medianBg, stdBg, varBg, pixelsBg, maskBg2D