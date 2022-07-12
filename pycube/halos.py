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
from astropy.convolution import Gaussian2DKernel

from pycube.core import manip

from IPython import embed

cSpeed = 299792.458  # in km/s


def smoothChiCube(dataCube,
                  statCube,
                  sSmooth=1.,
                  vSmooth=1.,
                  truncate=5.):
    """Given a (PSF subtacted) cube, the macro smooth both
    the dataCube and the statCube with a 3D gaussian Kernel.
    The same sigma is considered in both spatial axes
    (sSmooth), while a different one can be set for the
    spectral axis (vSmooth).
    Note that the 'STAT' cube is convolved as convol^2[statCube].
    The smoothChiCube is than created as:
                   convol[dataCube]
    sChiCube = ------------------------
               sqrt(convol**2[statCube])
    The chiCube is simply:
                       dataCube
    chiCube = -------------------------
                    sqrt(statCube)
    Given that scipy has hard time to deal with NaNs, NaNs
    values in the dataCube are converted to zeroes, while the
    NaNs in the statCube to nanmax(statCube).

    Parameters
    ----------
    dataCube: np.array
        data in a 3D array
    statCube : np.array
        variance in a 3D array
    sSmooth : float
        smooth length in pixel in the spatial direction
    vSmooth : float
        smooth length in pixel in the velocity direction
    truncate : float
        number of sigma after which the smoothing kernel
        is truncated

    Returns
    -------
    chiCube, sChiCube : np.array
        X-cube and smoothed X-cube
    """

    print("smoothChiCube: Smoothing Cube with 3D Gaussian Kernel")

    dataSigma = (vSmooth, sSmooth, sSmooth)
    statSigma = (vSmooth / np.sqrt(2.), sSmooth / np.sqrt(2.), sSmooth / np.sqrt(2.))

    # Removing nans
    dataCubeTmp = np.nan_to_num(dataCube, copy='True')
    statCubeTmp = np.copy(statCube)
    statCubeTmp[np.isnan(statCube)] = np.nanmax(statCube)

    # smooth cubes
    sDataCube = ndimage.filters.gaussian_filter(dataCubeTmp, dataSigma,
                                                truncate=truncate)
    sStatCube = ndimage.filters.gaussian_filter(statCubeTmp, statSigma,
                                                truncate=truncate)

    # Cleaning up memory
    del dataSigma
    del statSigma
    gc.collect()

    return dataCubeTmp / np.sqrt(statCubeTmp), sDataCube / np.sqrt(sStatCube)


def maskHalo(chiCube,
             xPix,
             yPix,
             zPix,
             rBadPix,
             rMaxPix,
             rConnect=2,
             threshold=2.,
             thresholdType='relative',
             badPixelMask=None,
             nSigmaExtreme = 5.,
             output='Object',
             debug=False,
             showDebug=False):
    """Given a PSF subtracted X cube (either smoothed or not) the
    macro, after masking some regions, performs a friends of friends
    search for connected pixels above a certain threshold in S/N.
    The first step is to identify the most significant voxel in
    proximity of the quasar position (given as xPix, yPix, zPix).
    The code assumes that the position of the extended halo is known
    and so started to create the mask of connected pixels from this
    point and from the most significant voxel within a spherical
    radius of 3.*rBadPix from (xPix, yPix, zPix).
    From there the macro searches for neighbor pixels that are above
    the threshold and creates a mask for the extended emission.

    Parameters
    ----------
    chiCube : np.array
        3D X cube output of smoothChiCube. This is constructed
        as data/noise (or as smooth(data)/smooth(noise)).
    xPix, yPix, zPix : float
        position from where start to search for the presence
        of a halo. This is the position of the quasar in x and y
        and the expected position of the extended halo at the
        quasar redshift in z.
    rBadPix : int
        radius of the circular region centred in xPix, yPix that
        will be masked. This is typically due to the absence of
        information at the quasar location caused by the normalization
        of the empirical PSF model to the central region.
    rMaxPix : int
        the circular region centred in xPix and yPix with radius
        rMaxPix will be masked and not considered in the search of
        extended emission. This helps to prevent boundary problems
        and to speed up the algorithm.
    rConnect : int
        default is rConnect=2. Connecting distance used in the FoF
        algorithm.
    threshold : float
        S/N threshold to consider a pixel as part of the extended
        emission. Default is 2.
    thresholdType : str
        'relative': A pixel will be considered as a part of a halo
        if it is above 'threshold' times the sigma of the distribution
        of S/N of the pixels.
        'absolute' : A pixel will be considered as a part of a halo
        if it is above the value of 'threshold'.
    badPixelMask : np.array, or mask of boolean
        2D mask to remove spatial pixels from the estimate of the halo
        location. If 1 (or True) the spatial pixel will be removed.
    output : string
        root file name for outputs

    Returns
    -------
    maskHalo : np.array
        mask where the detected extended emission is set to 1 and
        the background is set to 0. It has the same shape of the
        input chiCube.
    """

    # Creating a mask
    print("maskHalo: removing inner region around (x,y)=({:.2f},{:.2f})".format(float(xPix), float(yPix)))
    print("          with radius {} pixels".format(rBadPix))
    badMask = manip.location(chiCube[0, :, :], x_position=xPix, y_position=yPix,
                                     semi_maj=rBadPix, semi_min=rBadPix)

    print("maskHalo: removing outer region around (x,y)=({:.2f},{:.2f})".format(float(xPix), float(yPix)))
    print("          with radius {} pixels".format(rMaxPix))
    badMaskOuter = manip.location(chiCube[0, :, :], x_position=xPix, y_position=yPix,
                                          semi_maj=rMaxPix, semi_min=rMaxPix)
    badMask[(badMaskOuter == 0)] = 1
    del badMaskOuter

    if badPixelMask is not None:
        print("maskHalo: removing {} bad voxels".format(np.sum(badPixelMask)))
        badMask[(badPixelMask > 0)] = 1

    # Filling up mask
    chiCubeTmp = np.copy(chiCube)
    chiCubeMask = np.zeros_like(chiCube)
    channel_array = manip.channel_array(chiCubeTmp, 'z')
    zMax, yMax, xMax = np.shape(chiCube)
    for channel in channel_array:
        chiCubeTmp[channel, :, :][(badMask == 1)] = np.nan
        chiCubeMask[channel, :, :][(badMask == 1)] = 1

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

    chiCubeAve, chiCubeMed, chiCubeSig = manip.statFullCube(chiCubeTmp, nSigmaExtreme=nSigmaExtreme)
    chiCubeAveZ, chiCubeMedZ, chiCubeSigZ = manip.statFullCubeZ(chiCubeTmp, nSigmaExtreme=nSigmaExtreme)
    print("maskHalo: the median value of the voxels is: {:+0.4f}".format(chiCubeMed))
    print("          and the sigma is: {:+0.4f}".format(chiCubeSig))
    if thresholdType == 'relative':
        print("maskHalo: the average relative threshold value set to {:0.2f}*{:0.4f}={:0.4f}".format(threshold,
                                                                                                     chiCubeSig,
                                                                                                     threshold * chiCubeSig))
        thresholdHalo = threshold * chiCubeSigZ
    elif thresholdType == 'absolute':
        print("maskHalo: absolute threshold value set to {:0.2f}".format(threshold))
        thresholdHalo = threshold * np.ones_like(chiCubeSigZ)
    else:
        print("maskHalo: WARNING!")
        print("          no thresholdType set, assumed relative")
        print("maskHalo: the average relative threshold value set to {:0.2f}*{:0.4f}={:0.4f}".format(threshold,
                                                                                                     chiCubeSig,
                                                                                                     threshold * chiCubeSig))
        thresholdHalo = threshold * chiCubeSigZ

    if debug:
        print("maskHalo: Saving debug image on {}_voxelDistribution.pdf".format(output))
        print("          in principle the distribution should be gaussian")
        print("          showing only channel {}".format(np.int(zMax / 2.)))

        manip.nicePlot()

        # Populating the histogram
        chiCubeHist, chiCubeEdges = np.histogram(
            chiCubeTmp[np.int(zMax / 2.), :, :][np.isfinite(chiCubeTmp[1, :, :])].flatten(),
            bins="fd", density=True)
        # fitting of the histogram
        chiCubeEdgesBin = np.nanmedian(chiCubeEdges - np.roll(chiCubeEdges, 1))
        gaussBest, gaussCovar = curve_fit(manip.gaussian,
                                          chiCubeEdges[:-1],
                                          chiCubeHist,
                                          p0=[np.nansum(chiCubeHist * chiCubeEdgesBin) / (
                                                      chiCubeSig * np.sqrt((2. * np.pi))),
                                              chiCubeMed,
                                              chiCubeSig])

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axHist = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        # Plotting field image
        axImage.imshow(chiCubeTmp[np.int(zMax / 2.), :, :],
                       cmap="Greys", origin="lower",
                       vmin=chiCubeMed - 3. * chiCubeSig,
                       vmax=chiCubeMed + 3. * chiCubeSig)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_title(r"Channel {} X-image".format(int(zMax / 2.)))
        axImage.set_xlim(left=xPix - rMaxPix, right=xPix + rMaxPix)
        axImage.set_ylim(bottom=yPix - rMaxPix, top=yPix + rMaxPix)

        # Plotting pixel distribution
        axHist.step(chiCubeEdges[:-1], chiCubeHist, color="gray",
                    zorder=3)
        axHist.plot(chiCubeEdges[:-1], manip.gaussian(chiCubeEdges[:-1], *gaussBest),
                    color='black', zorder=2)
        axHist.axvline(chiCubeMed, color="black",
                       zorder=1, linestyle=':')
        axHist.set_xlim(left=chiCubeMed - 3. * chiCubeSig,
                        right=chiCubeMed + 3. * chiCubeSig)
        axHist.set_ylim(bottom=-0.01 * np.nanmax(manip.gaussian(chiCubeEdges[:-1], *gaussBest)),
                        top=1.2 * np.nanmax(manip.gaussian(chiCubeEdges[:-1], *gaussBest)))
        axHist.text(0.52, 0.9, "Median", transform=axHist.transAxes)
        axHist.set_ylabel(r"Pixel Distribution", size=30)
        axHist.set_xlabel(r"X", size=30)
        axHist.set_title(r"X values distribution")

        plt.tight_layout()
        plt.savefig(output + "_voxelDistribution.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del chiCubeHist
        del chiCubeEdges
        del gaussBest
        del gaussCovar

    # Searching for the Max value from which to start to look for connections
    print("maskHalo: searching for extended emission")
    print("          starting from from (x,y,z)=({:0.2f},{:0.3f},{:0.0f})".format(float(xPix), float(yPix),
                                                                                  float(zPix)))
    # Check small cube
    smallChiCubeTmp = chiCubeTmp[int(zPix - 3. * rBadPix):int(zPix + 3. * rBadPix),
                      int(yPix - 3. * rBadPix):int(yPix + 3. * rBadPix),
                      int(xPix - 3. * rBadPix):int(xPix + 3. * rBadPix)]
    if np.nansum(np.isfinite(smallChiCubeTmp)) > 0:
        maxSChi = np.nanmax(smallChiCubeTmp)
        # Selecting closest value
        zMaxSChi, yMaxSChi, xMaxSChi = np.where(chiCubeTmp == maxSChi)
        distPix = manip.distFromPixel(zPix, yPix, xPix,
                                        zMaxSChi, yMaxSChi, xMaxSChi)
        zMaxSChi = int(zMaxSChi[np.where(distPix == np.min(distPix))])
        yMaxSChi = int(yMaxSChi[np.where(distPix == np.min(distPix))])
        xMaxSChi = int(xMaxSChi[np.where(distPix == np.min(distPix))])
        print("maskHalo: the maximum S/N detected is {:0.3f} ".format(maxSChi))
        print("          at the location (x,y,z)=({},{},{})".format(xMaxSChi, yMaxSChi, zMaxSChi))
    else:
        maxSChi = threshold
        zMaxSChi, yMaxSChi, xMaxSChi = zPix, yPix, xPix
    del smallChiCubeTmp
    print("maskHalo: starting to fill the halo mask")
    # this mask is equal to 1 if the pixel is considered part of the halo
    # equal to 0 if it is not
    maskHalo = np.zeros_like(chiCube)
    if maxSChi > thresholdHalo[zMaxSChi]:
        maskHalo[zMaxSChi, yMaxSChi, xMaxSChi] = 1
    # The code also generate a 'seed' at the expected position of the
    # extended halo. This is helpful to avoid to be too dependent on the
    # location of the brightest pixel. It will be removed before output.
    zSeedMin, zSeedMax = int(zPix - 5. * rConnect), int(zPix + 5. * rConnect)
    ySeedMin, ySeedMax = int(yPix - rBadPix), int(yPix + rBadPix)
    xSeedMin, xSeedMax = int(xPix - rBadPix), int(xPix + rBadPix)
    maskHalo[zSeedMin:zSeedMax, ySeedMin:ySeedMax, xSeedMin:xSeedMax] = 1
    chiCubeMask[zSeedMin:zSeedMax, ySeedMin:ySeedMax, xSeedMin:xSeedMax] = 0

    # here the code start to propagate the mask in the neighbour pixels
    nConnectedPixels = 0
    nConnectedPixelsNew = int(np.nansum(maskHalo))
    while nConnectedPixels < nConnectedPixelsNew:
        nConnectedPixels = int(np.nansum(maskHalo))
        # create a new mask around the identified voxels
        maskHaloTemp = np.copy(maskHalo)
        zMask, yMask, xMask = np.where(maskHaloTemp == 1)
        for zMaskTemp, yMaskTemp, xMaskTemp in zip(zMask, yMask, xMask):
            zMaskMin, zMaskMax = int(zMaskTemp - rConnect), int(zMaskTemp + rConnect)
            yMaskMin, yMaskMax = int(yMaskTemp - rConnect), int(yMaskTemp + rConnect)
            xMaskMin, xMaskMax = int(xMaskTemp - rConnect), int(xMaskTemp + rConnect)
            maskHaloTemp[zMaskMin:zMaskMax, yMaskMin:yMaskMax, xMaskMin:xMaskMax] = 1
        # check that voxels in maskHaloTemp are above the threshold
        newChiCube = np.copy(chiCube)
        newChiCube[~np.isfinite(newChiCube)] = 0.
        newChiCube[(chiCubeMask == 1)] = 0.
        newChiCube *= maskHaloTemp.astype(float)
        for channel in np.arange(0, zMax):
            maskHalo[channel, :, :][newChiCube[channel, :, :] > thresholdHalo[channel]] = 1
        nConnectedPixelsNew = int(np.nansum(maskHalo))
    # Removing seed
    maskHalo[zSeedMin:zSeedMax, ySeedMin:ySeedMax, xSeedMin:xSeedMax] = 0
    # Removing masked data

    if debug:
        print("maskHalo: Creating debug image")
        print("          Plotting Channel {} where the most significant voxel is.".format(zMaxSChi))
        print("          The location of this voxel is marked with a red circle")
        print("          The position of the quasars is in blue")

        manip.nicePlot()

        maskHalo2D, maskHaloMinZ, maskHaloMaxZ = spectralMaskHalo(maskHalo)

        plt.figure(1, figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        axImage = plt.subplot2grid((1, 2), (0, 0), colspan=1)
        axMask = plt.subplot2grid((1, 2), (0, 1), colspan=1)

        xPlotMin, xPlotMax = int(xPix - rMaxPix), int(xPix + rMaxPix)
        yPlotMin, yPlotMax = int(yPix - rMaxPix), int(yPix + rMaxPix)
        stdPlot = np.nanstd(chiCube[zMaxSChi,
                            yPlotMin:xPlotMax,
                            xPlotMin:yPlotMax])
        # Plotting field image
        axImage.imshow(chiCube[zMaxSChi, :, :],
                       cmap="Greys", origin="lower",
                       vmin=-1. * stdPlot,
                       vmax=+3. * stdPlot)
        axMask.contour(maskHalo[zMaxSChi, :, :], colors='maroon',
                       alpha=0.9, origin="lower", linewidths=0.5)
        axMask.contour(maskHalo2D, colors='orangered',
                       alpha=0.5, origin="lower", linewidths=0.5)
        axImage.set_xlabel(r"X [Pixels]", size=30)
        axImage.set_ylabel(r"Y [Pixels]", size=30)
        axImage.set_xlim(left=xPlotMin, right=xPlotMax)
        axImage.set_ylim(bottom=yPlotMin, top=yPlotMax)
        axImage.set_title(r"Channel {} Map".format(zMaxSChi))

        maxArtist = Ellipse(xy=(xMaxSChi, yMaxSChi),
                            width=rConnect,
                            height=rConnect,
                            angle=0.)
        maxArtist.set_facecolor("none")
        maxArtist.set_edgecolor("red")
        maxArtist.set_alpha(0.5)
        axImage.add_artist(maxArtist)

        qsoArtist = Ellipse(xy=(xPix, yPix),
                            width=rBadPix,
                            height=rBadPix,
                            angle=0.)
        qsoArtist.set_facecolor("none")
        qsoArtist.set_edgecolor("blue")
        qsoArtist.set_alpha(0.5)
        axImage.add_artist(qsoArtist)

        # Plotting mask image
        axMask.imshow(badMask,
                      cmap="Greys", origin="lower",
                      vmin=0.,
                      vmax=.5)
        axMask.contour(maskHalo[zMaxSChi, :, :], colors='maroon',
                       alpha=0.9, origin="lower", linewidths=0.5)
        axMask.contour(maskHalo2D, colors='orangered',
                       alpha=0.5, origin="lower", linewidths=0.5)
        axMask.set_xlabel(r"X [Pixels]", size=30)
        axMask.set_ylabel(r"Y [Pixels]", size=30)
        axMask.set_xlim(left=xPix - rMaxPix, right=xPix + rMaxPix)
        axMask.set_ylim(bottom=yPix - rMaxPix, top=yPix + rMaxPix)
        axMask.set_title(r"Excluded Pixels Mask")

        print("maskHalo: debug image saved in {}_maskHaloStartingVoxel.pdf".format(output))
        plt.tight_layout()
        plt.savefig(output + "_maskHaloStartingVoxel.pdf", dpi=400.,
                    format="pdf", bbox_inches="tight")
        if showDebug:
            plt.show()
        plt.close()
        del maskHalo2D
        del maskHaloMinZ
        del maskHaloMaxZ

    # Clearing up memory
    del badMask
    del newChiCube
    del chiCubeTmp
    del chiCubeMask

    return maskHalo


def spectralMaskHalo(maskHalo):
    """Given the halo mask, the macro returns
    a 2D mask in x and y and the min and max
    channel where the halo is detected in the z-axis.

    Parameters
    ----------
    maskHalo : np.array
        3D mask of the halo location.

    Returns
    -------
    maskHalo2D : np.array
        2D mask of the halo location (collapsed along the
        z-axis).
    maskHaloMinZ, maskHaloMaxZ : int
        min and max channel where the halo is detected along
        the z-axis
    """

    print("spectralMaskHalo: collapsing halo mask")
    if (np.nansum(maskHalo) < 2):
        print("spectralMaskHalo: not enough voxels in the mask")
        zMax, yMax, xMax = np.shape(maskHalo)
        return np.zeros_like(maskHalo[0, :, :], int), 0, zMax

    maskHalo2D = np.nansum(maskHalo, axis=0)
    maskHalo2D[(maskHalo2D > 0)] = 1

    # Collapsing along 1,2 to obtain the z-map
    maskHaloZ = np.nansum(maskHalo, axis=(1, 2))
    zMax, yMax, xMax = np.shape(maskHalo)
    channel = np.arange(0, zMax, 1, int)
    maskHaloMinZ, maskHaloMaxZ = np.nanmin(channel[maskHaloZ > 0]), np.nanmax(channel[maskHaloZ > 0])

    # Cleaning up memory
    del maskHaloZ
    del channel

    return maskHalo2D, maskHaloMinZ, maskHaloMaxZ


def cleanMaskHalo(maskHalo, deltaZMin=2, minGood=100, channelMin=None, channelMax=None, debug=False, showDebug=False):
    """
    Given the halo mask, the macro performs some quality
    check:
    * If a spatial pixel (x,y) has less than deltaZmin consecutive voxels identified along the z-axis, this will
    be removed from the mask
    * If the total number of voxels is less than minGood the halo is considered as not detected and the mask is cleaned.

    Parameters
    ----------
    maskHalo : np.array
        3D mask of the halo location.
    deltaZMin : int
        min size in the spectral axis to consider the voxel
        as part of the halo
    channelMin, channelMax : np.array
        only voxels between channelMin and channelMax in the
        spectral direction will be considered in the creation
        of the cleanMask

    Returns
    -------
    maskHaloClean : np.array
        cleaned 3D mask of the halo location.
    """

    print("cleanMaskHalo: cleaning halo mask")

    if np.nansum(maskHalo) < np.int(minGood / 2.):
        print("cleanMaskHalo: not enough voxels in the mask")
        return np.zeros_like(maskHalo, int)

    maskHaloClean = np.copy(maskHalo)
    # Collapsing along 1,2 to obtain the z-map
    maskHaloCleanXY = np.nansum(maskHaloClean, axis=0)
    zMax, yMax, xMax = np.shape(maskHaloClean)
    for channel in np.arange(0, zMax, 1, int):
        maskHaloClean[channel, :, :][(maskHaloCleanXY <= deltaZMin)] = 0
    if np.nansum(maskHaloClean) <= minGood:
        maskHaloClean[:, :, :] = 0
    if channelMin is not None:
        maskHaloClean[0:channelMin, :, :] = 0
    if channelMax is not None:
        maskHaloClean[channelMax:-1, :, :] = 0

    print("cleanMaskHalo: Removed {} voxels from the mask".format(np.sum(maskHalo) - np.sum(maskHaloClean)))

    if debug:
        zMax, yMax, xMax = np.shape(maskHalo)
        channelY = np.arange(0, yMax, 1, int)
        channelX = np.arange(0, xMax, 1, int)
        maskHaloY = np.nansum(maskHalo, axis=(0, 2))
        maskHaloX = np.nansum(maskHalo, axis=(0, 1))
        maskHaloMinY, maskHaloMaxY = np.nanmin(channelY[maskHaloY > 0]), np.nanmax(channelY[maskHaloY > 0])
        maskHaloMinX, maskHaloMaxX = np.nanmin(channelX[maskHaloX > 0]), np.nanmax(channelX[maskHaloX > 0])
        maskHalo2D, _, _ = spectralMaskHalo(maskHalo)
        maskHaloClean2D, _, _ = spectralMaskHalo(maskHaloClean)
        plt.figure(1, figsize=(6, 6))
        plt.contour(maskHalo2D, colors='blue',
                    alpha=0.9, origin="lower")
        plt.contour(maskHaloClean2D, colors='red',
                    alpha=0.9, origin="lower")
        plt.xlim(maskHaloMinX - 2, maskHaloMaxX + 2)
        plt.ylim(maskHaloMinY - 2, maskHaloMaxY + 2)
        plt.xlabel(r"X [Pixels]", size=30)
        plt.ylabel(r"Y [Pixels]", size=30)
        if showDebug:
            plt.show()
        plt.close()
        del maskHalo2D
        del maskHaloClean2D
        del maskHaloY
        del maskHaloX
    # Cleaning up memory
    del maskHaloCleanXY

    return maskHaloClean


def makeMoments(headCube,
                dataCube,
                statCube,
                maskHalo,
                centralWave=None,
                sSmooth=None,
                truncate=5.,
                debug=False,
                showDebug=False):
    """ Given a PSF-Subtracted datacube, this macro extracts the moment 0, 1, 2
    maps of the halo identified by maskHalo.
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
    headCube : hdu header
        fits header for the cube
    dataCube : np.array
        data in a 3D array
    statCube : np.array
        variance in a 3D array
    maskHalo : np.array
        mask where the detected extended emission is set to 1 and the background is set to 0. It has the same shape
        of the input dataCube.
    centralWave : float
        wavelength in Ang. from which to calculate the
        velocity shifts. If None, the macro will calculate
        it from the spectrum of the halo.
    sSmooth : float
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

    print("makeMoments: estimating halo moments")

    if (np.nansum(maskHalo) < 3):
        print("makeMoments: not enough voxels to calculate moment maps")
        centralWave = 0.
        nullImage = np.zeros_like(dataCube[0, :, :], float)
        return nullImage, nullImage, nullImage, centralWave

    dataCubeTmp = np.copy(dataCube)

    gkernelRadius = 0
    while np.nansum(~np.isfinite(dataCubeTmp)) > 0:
        gkernelRadius = gkernelRadius + 2
        print("makeMoments: masking {} NaNs with a {} spatial pixel Gaussian Kernel".format(
            np.nansum(~np.isfinite(dataCubeTmp)), gkernelRadius))
        print("             the total number of voxels is {}".format(np.size(dataCubeTmp)))
        gkernel = astropy.convolution.Gaussian2DKernel(gkernelRadius)
        zMax, yMax, xMax = np.shape(dataCube)
        for channel in np.arange(0, zMax):
            dataChannel = np.copy(dataCubeTmp[channel, :, :])
            # correct for nan with blurred images
            blurChannel = astropy.convolution.convolve(np.copy(dataCube[channel, :, :]), gkernel)
            dataChannel[np.logical_not(np.isfinite(dataChannel))] = blurChannel[
                np.logical_not(np.isfinite(dataChannel))]
            dataCubeTmp[channel, :, :] = dataChannel[:, :]
        del blurChannel, dataChannel

    if (sSmooth is not None):
        print("makeMoments: smoothing cube")
        vSmooth = 0
        dataSigma = (vSmooth, sSmooth, sSmooth)
        # Removing nans
        dataCubeTmp = np.nan_to_num(dataCube, copy=True)
        dataCubeTmp = ndimage.filters.gaussian_filter(dataCubeTmp, dataSigma,
                                                      truncate=truncate)
    # Extract Spectrum
    maskHalo2D, maskHaloMinZ, maskHaloMaxZ = spectralMaskHalo(maskHalo)
    fluxHalo, errFluxHalo = manip.quickSpectrumNoBgMask(dataCube,
                                                          statCube,
                                                          maskHalo2D)
    fluxHaloSmooth, errFluxHaloSmooth = manip.quickSpectrumNoBgMask(dataCubeTmp,
                                                                      statCube,
                                                                      maskHalo2D)

    # removing voxels outside the halo
    dataCubeTmp[(maskHalo < 1)] = np.float(0.)

    # Defining wavelength range
    zMax, yMax, xMax = np.shape(dataCube)
    channels = np.arange(0, zMax)
    wave = manip.convert_to_wave(headCube, channels)

    # find central wavelength of the halo:
    # Extract "optimally extracted" spectrum
    fluxHaloOpt = np.nansum(dataCubeTmp, axis=(1, 2))

    if centralWave is None:
        centralWave = np.nansum(wave * fluxHaloOpt) / np.nansum(fluxHaloOpt)
    print("makeMoments: the central wavelength considered is {:+.2f}".format(centralWave))

    # Moment Zero
    mom0 = manip.collapse_cube(dataCubeTmp)

    # Moment One
    velShift = (wave - centralWave) / centralWave * cSpeed  # in km/s
    velShiftCube = np.zeros_like(dataCubeTmp, float)
    for channel in np.arange(0, zMax):
        velShiftCube[channel, :, :] = velShift[channel]
    mom1 = np.nansum(velShiftCube * dataCubeTmp, axis=0) / np.nansum(dataCubeTmp, axis=0)

    # Moment Two
    sigmaCube = np.ones_like(dataCubeTmp, float)
    for channel in np.arange(0, zMax):
        sigmaCube[channel, :, :] = np.power(velShift[channel] - mom1[:, :], 2.)
    mom2 = np.sqrt(np.nansum(sigmaCube * dataCubeTmp, axis=0) / np.nansum(dataCubeTmp, axis=0))

    if debug:

        print("makeMoments: showing debug image:")
        print("             spectrum and optimally extracted spectrum")

        manip.nicePlot()

        # Setting limits for halo image
        channelY = np.arange(0, yMax, 1, int)
        channelX = np.arange(0, xMax, 1, int)
        maskHaloY = np.nansum(maskHalo, axis=(0, 2))
        maskHaloX = np.nansum(maskHalo, axis=(0, 1))
        maskHaloMinY, maskHaloMaxY = np.nanmin(channelY[maskHaloY > 0]), np.nanmax(channelY[maskHaloY > 0])
        maskHaloMinX, maskHaloMaxX = np.nanmin(channelX[maskHaloX > 0]), np.nanmax(channelX[maskHaloX > 0])

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        axImag = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        axSpec = plt.subplot2grid((1, 3), (0, 1), colspan=2)

        axImag.imshow(mom0,
                      cmap="Greys", origin="lower")
        axImag.set_xlim(left=maskHaloMinX, right=maskHaloMaxX)
        axImag.set_ylim(bottom=maskHaloMinY, top=maskHaloMaxY)
        axImag.set_xlabel(r"X [Pixels]", size=30)
        axImag.set_ylabel(r"Y [Pixels]", size=30)

        axSpec.plot(wave, fluxHalo, color='black', zorder=3, label='Total')
        axSpec.plot(wave, fluxHaloSmooth, color='gray', zorder=3, label='Smooth')
        axSpec.plot(wave, fluxHaloOpt, color='red', zorder=3, label='Opt. Ex.')
        axSpec.plot(wave, errFluxHalo, color='gray', alpha=0.5, zorder=2, label='Error')
        axSpec.legend(loc='upper left', ncol=2, fontsize=15)
        axSpec.set_xlabel(r"Wavelength", size=30)
        axSpec.set_ylabel(r"Flux", size=30)
        axSpec.axvline(centralWave, color="red",
                       zorder=1, linestyle='-')
        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()

        print("makeMoments: showing debug image:")
        print("             0, 1, and 2 moments")

        plt.figure(1, figsize=(18, 6))
        gs = gridspec.GridSpec(1, 3)

        axMom0 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
        axMom1 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
        axMom2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)

        imgMom0 = axMom0.imshow(mom0,
                                cmap="Greys", origin="lower")
        axMom0.set_xlim(left=maskHaloMinX, right=maskHaloMaxX)
        axMom0.set_ylim(bottom=maskHaloMinY, top=maskHaloMaxY)
        axMom0.set_xlabel(r"X [Pixels]", size=30)
        axMom0.set_ylabel(r"Y [Pixels]", size=30)
        cbMom0 = plt.colorbar(imgMom0, ax=axMom0, shrink=0.5)

        imgMom1 = axMom1.imshow(mom1,
                                cmap="Greys", origin="lower",
                                vmin=-0.5 * np.nanmax(mom1),
                                vmax=+0.5 * np.nanmax(mom1))
        axMom1.set_xlim(left=maskHaloMinX, right=maskHaloMaxX)
        axMom1.set_ylim(bottom=maskHaloMinY, top=maskHaloMaxY)
        axMom1.set_xlabel(r"X [Pixels]", size=30)
        axMom1.set_ylabel(r"Y [Pixels]", size=30)
        cbMom1 = plt.colorbar(imgMom1, ax=axMom1, shrink=0.5)

        imgMom2 = axMom2.imshow(mom2,
                                cmap="Greys", origin="lower",
                                vmin=0.,
                                vmax=+0.9 * np.nanmax(mom2))
        axMom2.set_xlim(left=maskHaloMinX, right=maskHaloMaxX)
        axMom2.set_ylim(bottom=maskHaloMinY, top=maskHaloMaxY)
        axMom2.set_xlabel(r"X [Pixels]", size=30)
        axMom2.set_ylabel(r"Y [Pixels]", size=30)
        cbMom2 = plt.colorbar(imgMom2, ax=axMom2, shrink=0.5)

        plt.tight_layout()
        if showDebug:
            plt.show()
        plt.close()

    # Cleaning up memory
    del maskHalo2D, maskHaloMinZ, maskHaloMaxZ
    del dataCubeTmp
    del fluxHalo, errFluxHalo
    del fluxHaloSmooth, errFluxHaloSmooth
    del wave
    del velShiftCube, sigmaCube
    gc.collect()

    return mom0, mom1, mom2, centralWave


def maxExtent(maskHalo2D, xPix=None, yPix=None):
    """Given a 2D mask halo, the macro calculates
    the maximum extend of the halo in the spatial
    direction and the maximum distance from xPix,
    yPix (if given).

    Parameters
    ----------
    maskHalo2D : np.array
        2D mask of the halo location

    Returns
    -------
    maxExtent : np.float
        maximum extent in pixels
    fromPixExtent : np.float
        maximum extent from (xPix, yPix) in pixels
    """

    print("maxExtent: estimating maximal extent")

    if (np.nansum(maskHalo2D) < 3):
        print("maxExtent: not enough spatial pixels")
        return 0., 0.

    maxExtent = 0.

    yPivot, xPivot = np.where(maskHalo2D > 0)
    zPivot = np.zeros_like(yPivot, int)
    for zIdx, yIdx, xIdx in zip(zPivot, yPivot, xPivot):
        maxExtentTemp = np.nanmax(manip.distFromPixel(zIdx, yIdx, xIdx, zPivot, yPivot, xPivot))
        if (maxExtentTemp > maxExtent):
            maxExtent = maxExtentTemp

    if xPix is None:
        return float(maxExtent), 0.

    fromPixExtent = np.nanmax(manip.distFromPixel(float(0.), yPix, xPix, zPivot, yPivot, xPivot))

    return float(maxExtent), float(fromPixExtent)


