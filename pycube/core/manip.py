import numpy as np
import sep

def convert_to_wave(datacube, channels):
    wave = datacube.header['CRVAL3'] + (np.array(channels) * datacube.header['CD3_3'])
    return np.array(wave, dtype=float)


def collapse_cube(datacube, min_lambda, max_lambda):
    """
    Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    Inputs:
        datacube: 3D data file
        min_lambda: minimum wavelength
        max_lambda: maximum wavelength
    Returns:
        col_cube: Condensed 2D array of 3D file.
    """
    if datacube is None:
        print("Object passed is None. Returning object..")
        return None
    datacopy = np.copy(datacube)
    z_max, y_max, x_max = np.shape(datacopy)

    if max_lambda > z_max:
        max_lambda = z_max
        print("Exceed wavelength in datacube. Max value is set to {}".format(int(z_max)))
    if min_lambda < 0:
        min_lambda = 0
        print("Invalid wavelength value for min. Min value is set to 0")

    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :])
    return col_cube

#working on setting this to be user defined.
def findSources(self,
                sigDetect=3.501,
                minArea=16.):

    print("findSources: Starting sources detection")
    print("findSources: Creating background model")

    bgMedian = np.nanmedian(self.colData)
    bgSigma = np.sqrt(np.nanmedian(self.colStat))
    bgMask = np.zeros_like(self.colData)

    bgMask[(np.abs(self.colData - bgMedian) > 7. * bgSigma)] = int(1)
    imgBg = sep.Background(self.colData, mask=bgMask,
                            bw=64., bh=64., fw=5., fh=5.)

    imgDataNoBg = np.copy(self.colData) - imgBg
    print("findSources: Searching sources {}-sigma above noise".format(sigDetect))
    allObjects = sep.extract(imgDataNoBg, sigDetect,
                            var=self.colStat,
                            minarea=minArea,
                            filter_type='matched',
                            gain=1.1,
                            clean=True,
                            deblend_cont=0.3,
                            filter_kernel=None)
    # Sorting sources by flux at the peak
    indexByFlux = np.argsort(allObjects['peak'])[::-1]
    allObjects = allObjects[indexByFlux]
    goodSources = allObjects['flag'] < 1
    xPix = np.array(allObjects['x'][goodSources])
    yPix = np.array(allObjects['y'][goodSources])
    aPix = np.array(allObjects['a'][goodSources])
    bPix = np.array(allObjects['b'][goodSources])
    angle = np.array(allObjects['theta'][goodSources]) * 180. / np.pi
    print("findSources: {} good sources detected".format(np.size(xPix)))
    # Deleting temporary images to clear up memory
    del imgBg
    del imgDataNoBg
    del goodSources
    print(xPix, yPix, aPix, bPix, angle, allObjects)

def source(datacube, ra, dec, z):
    """

    Args:
        datacube:
        ra:
        dec:
        z:

    Returns:

    """



