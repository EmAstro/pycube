import os
import numpy as np
from photutils.aperture import CircularAperture, CircularAnnulus,aperture_photometry
from astropy import wcs
from mpdaf.obj import Cube, Spectrum,WaveCoord
from astropy.io import fits
from astropy import wcs
import pyregion
import matplotlib.pyplot as plt
from lmfit import Model
import shutil
from scipy.stats import norm
from scipy.optimize import curve_fit

print("Creating Main Directory in Desktop")
directory = "apertures"
desktop_path = os.path.expanduser('~') + '/Desktop'
dir_path = os.path.join(desktop_path, directory)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
else:
    shutil.rmtree(
        dir_path)  # Removes all the subdirectories! (use for cleaning up the old directory and create a new on)
    os.makedirs(dir_path)
print(f"DETECTION DIRECTORY CREATED\n")

print("histogram_and_fit_directory_path")
histogram_and_fit_directory_path = os.path.join(dir_path, "histogram_and_fit")
if not os.path.exists(histogram_and_fit_directory_path):
    os.mkdir(histogram_and_fit_directory_path)
else:
    shutil.rmtree(histogram_and_fit_directory_path)
    os.mkdir(histogram_and_fit_directory_path)
print("LSDcat DIRECTORY CREATED\n")

print("aperture_ds9_regions")
aperture_ds9_regions_path = os.path.join(dir_path, "aperture_ds9_regions")
if not os.path.exists(aperture_ds9_regions_path):
    os.mkdir(aperture_ds9_regions_path)
else:
    shutil.rmtree(aperture_ds9_regions_path)
    os.mkdir(aperture_ds9_regions_path)
print("LSDcat DIRECTORY CREATED\n")


zap ="/home/sai/Desktop/P183p05_DATACUBE_ZAP.fits"
cube = Cube(zap)
f = fits.open(zap)
#hdul = fits.open(cat)  # LSDcat CATALOGUE
w1 = wcs.WCS(f[1].header)
hdr = f[1].header  # to get data header info
WAVE_SCALE=1.25

mu_total_100=[]
std_total_100=[]

mu_total_200=[]
std_total_200=[]

mu_total_500=[]
std_total_500=[]

mu_total_1000=[]
std_total_1000=[]


def quickApPhotmetryNoBg(imgData,imgStat,xObj,yObj,rObj):
    if np.size(xObj) == 1:
        xObject = np.array([xObj])
        yObject = np.array([yObj])
    else:
        xObject = np.array(xObj)
        yObject = np.array(yObj)

    if np.size(rObj) == 1:
        rAperture = np.full_like(xObject, rObj, dtype=np.float_)
    else:
        rAperture = np.array(rObj)

    fluxObj = np.zeros_like(xObject, dtype=np.float_)
    errFluxObj = np.zeros_like(xObject, dtype=np.float_)

    for idxObj in range(0, np.size(xObject)):
        posObj = [xObject[idxObj], yObject[idxObj]]
        circObj = CircularAperture(posObj, r=rAperture[idxObj])
        apPhot = aperture_photometry(imgData, circObj)
        varApPhot = aperture_photometry(imgStat, circObj)
        fluxObj[idxObj] = apPhot['aperture_sum'][0]
        errFluxObj[idxObj] = np.power(np.array(varApPhot['aperture_sum'][0]), 0.5)

    del rAperture
    del posObj
    del circObj
    del apPhot
    del varApPhot

    return fluxObj, errFluxObj




def collapseCube(dataCube,statCube=None,minChannel=None,maxChannel=None,maskZ=None,toFlux=True):
    if toFlux:
        scaleFactor = WAVE_SCALE
        print("collapseCube: Output converted to erg/s/cm**2.")
        print("              (i.e. multiplied by {} Ang.)".format(scaleFactor))
    else:
        scaleFactor = 1.

    tempDataCube = np.copy(dataCube)
    zMax, yMax, xMax = np.shape(tempDataCube)
    if statCube is not None:
        tempStatCube = np.copy(statCube)

    if (minChannel is not None) & (maxChannel is not None):
        minChannelSort = np.min([minChannel, maxChannel])
        maxChannelSort = np.max([minChannel, maxChannel])
        minChannel, maxChannel = minChannelSort, maxChannelSort
        del minChannelSort
        del maxChannelSort
    if minChannel is None:
        print("collapseCube: minChannel set to 0")
        minChannel = np.int(0)
    if maxChannel is None:
        print("collapseCube: maxChannel set to {}".format(np.int(zMax)))
        maxChannel = np.int(zMax)
    if minChannel < 0:
        print("collapseCube: Negative value for minChannel set to 0")
        minChannel = np.int(0)
    if maxChannel > (zMax + 1):
        print("collapseCube: maxChannel is outside the cube size. Set to {}".format(np.int(zMax)))
        maxChannel = np.int(zMax)
    if maskZ is not None:
        tempDataCube[maskZ, :, :] = np.nan
        if statCube is not None:
            tempStatCube[maskZ, :, :] = np.nan

    collapsedDataImage = np.nansum(tempDataCube[minChannel:maxChannel, :, :] * scaleFactor, axis=0)
    if statCube is not None:
        collapsedStatImage = np.nansum(tempStatCube[minChannel:maxChannel, :, :] * scaleFactor * scaleFactor,
                                       axis=0)  # *np.sqrt((maxChannel-minChannel))/1.25/1.25
    else:
        print("collapseCube: Stat image will be created from a rough")
        print("              estimate of the background noise")
        # This is to remove brightes sources.
        medImage = np.nanmedian(collapsedDataImage)
        stdImage = np.nanstd(collapsedDataImage - medImage)
        collapsedStatImage = np.zeros_like(collapsedDataImage) + np.nanvar(
            collapsedDataImage[(collapsedDataImage - medImage) < (5. * stdImage)])
        del medImage
        del stdImage
    print("collapseCube: Images produced")

    # Deleting temporary cubes to clear up memory
    del tempDataCube
    if statCube is not None:
        del tempStatCube
    del scaleFactor

    return collapsedDataImage, collapsedStatImage




for z in range(1):
    print(f"run {z+1}/100")
    ra_random_pix = np.random.randint(20, hdr['NAXIS1'] - 20, 600)
    dec_random_pix = np.random.randint(20, hdr['NAXIS1'] - 20, 600)


    ra_deg = []
    dec_deg = []
    positions = []

    for i in range(len(ra_random_pix)):
        t = w1.all_pix2world([[ra_random_pix[i], dec_random_pix[i], 0]], 0)
        ra_deg.append(t[0][0])
        dec_deg.append(t[0][1])
        positions.append((ra_random_pix[i], dec_random_pix[i]))

    ra_deg = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)

    ra_reject = []
    dec_reject = []

    ra_one_accept = []
    dec_one_accept = []
    ra_one_accept_pix = []
    dec_one_accept_pix = []

    ra_two_accept = []
    dec_two_accept = []
    ra_two_accept_pix = []
    dec_two_accept_pix = []

    ra_three_accept = []
    dec_three_accept = []
    ra_three_accept_pix = []
    dec_three_accept_pix = []


    b_1 = pyregion.open('/home/sai/Desktop/Detection_pycube/LSDcat/LSDcat_possible_source_file.reg')


    for o in range(len(ra_deg)):
        for p in range(len(b_1)):
            if b_1[p].coord_list[0] - (b_1[0].coord_list[2] + 0.0001722) < ra_deg[o] < b_1[p].coord_list[0] + (
                    b_1[0].coord_list[2] + 0.0001722) and b_1[p].coord_list[1] - (b_1[0].coord_list[2] + 0.0001722) < \
                    dec_deg[o] < b_1[p].coord_list[1] + (b_1[0].coord_list[2] + 0.0001722):
                ra_reject.append(ra_deg[o])
                dec_reject.append(dec_deg[o])
                break
        else:
            ra_one_accept.append(ra_deg[o])
            dec_one_accept.append(dec_deg[o])
            ra_one_accept_pix.append(ra_random_pix[o])
            dec_one_accept_pix.append(dec_random_pix[o])

    ra_one_accept = np.asarray(ra_one_accept)
    dec_one_accept = np.asarray(dec_one_accept)
    ra_one_accept_pix = np.asarray(ra_one_accept_pix)
    dec_one_accept_pix = np.asarray(dec_one_accept_pix)

    b_2 = pyregion.open('/home/sai/Desktop/boxs_wcs.reg')
    for o_2 in range(len(ra_one_accept)):
        for p_2 in range(len(b_2)):
            if b_2[p_2].coord_list[0] - (b_2[p_2].coord_list[2] / 2) < ra_one_accept[o_2] < b_2[p_2].coord_list[0] + (
                    b_2[p_2].coord_list[2] / 2) and b_2[p_2].coord_list[1] - (b_2[p_2].coord_list[3] / 2) < \
                    dec_one_accept[o_2] < b_2[p_2].coord_list[1] + (b_2[p_2].coord_list[3] / 2):
                ra_reject.append(ra_one_accept[o_2])
                dec_reject.append(dec_one_accept[o_2])
                break
        else:
            ra_two_accept.append(ra_one_accept[o_2])
            dec_two_accept.append(dec_one_accept[o_2])
            ra_two_accept_pix.append(ra_one_accept_pix[o_2])
            dec_two_accept_pix.append(dec_one_accept_pix[o_2])

    ra_two_accept = np.asarray(ra_two_accept)
    dec_two_accept = np.asarray(dec_two_accept)
    ra_two_accept_pix = np.asarray(ra_two_accept_pix)
    dec_two_accept_pix = np.asarray(dec_two_accept_pix)

    #print(f"hi : {len(ra_two_accept)}")
    for o_3 in range(len(ra_two_accept)):
        for p_3 in range(len(ra_two_accept)):
            if ra_two_accept[p_3] - (0.0001722 * 1.5) < ra_two_accept[o_3] < ra_two_accept[p_3] + (0.0001722 * 1.5) and \
                    dec_two_accept[p_3] - (0.0001722 * 1.5) < dec_two_accept[o_3] < dec_two_accept[p_3] + (
                    0.0001722 * 1.5) and ra_two_accept[o_3] != ra_two_accept[p_3]:
                ra_reject.append(ra_two_accept[o_3])
                dec_reject.append(dec_two_accept[o_3])
                break
            elif ra_two_accept[o_3] == ra_two_accept[p_3]:
                continue
        else:
            ra_three_accept.append(ra_two_accept[o_3])
            dec_three_accept.append(dec_two_accept[o_3])
            ra_three_accept_pix.append(ra_two_accept_pix[o_3])
            dec_three_accept_pix.append(dec_two_accept_pix[o_3])

    ra_three_accept = np.asarray(ra_three_accept)
    dec_three_accept = np.asarray(dec_three_accept)
    ra_three_accept_pix = np.asarray(ra_three_accept_pix)
    dec_three_accept_pix = np.asarray(dec_three_accept_pix)

    #print("******************")
    #print(f"hi : {len(ra_two_accept)}")
    #print(len(ra_three_accept))
    name_of_region_file = os.path.join(aperture_ds9_regions_path, f"apertures_run_{z+1}.reg")
    file_3 = open(name_of_region_file, "w+")
    file_3.write("#Region file format: DS9 version 4.1 "
                 "\nglobal color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 "
                 "\nfk5\n")
    for i in range(len(ra_three_accept)):
        file_3.write(f"circle({ra_three_accept[i]},{dec_three_accept[i]},{0.0001722})\n")
    file_3.close()
    print("FILE CREATED\n")

    positions = []
    for i in range(len(ra_three_accept)):
        positions.append((ra_three_accept_pix[i], dec_three_accept_pix[i]))


    delta_lbda_100 = int((1e2 / 3e5) * 9001.)
    delta_lbda_200 = int((2e2 / 3e5) * 9001.)
    delta_lbda_500 = int((5e2 / 3e5) * 9001.)
    delta_lbda_1000 = int((1e3 / 3e5) * 9001.)
    lbda_range_100 = np.asarray([9001. - delta_lbda_100, 9001. + delta_lbda_100])
    lbda_range_200 = np.asarray([9001. - delta_lbda_200, 9001. + delta_lbda_200])
    lbda_range_500 = np.asarray([9001. - delta_lbda_500, 9001. + delta_lbda_500])
    lbda_range_1000 = np.asarray([9001. - delta_lbda_1000, 9001. + delta_lbda_1000])

    #print(cube.info())
    img_narrow_band_100 = cube.get_image((lbda_range_100[0], lbda_range_100[1]), method='sum')
    img_narrow_band_200 = cube.get_image((lbda_range_200[0], lbda_range_200[1]), method='sum')
    img_narrow_band_500 = cube.get_image((lbda_range_500[0], lbda_range_500[1]), method='sum')
    img_narrow_band_1000 = cube.get_image((lbda_range_1000[0], lbda_range_1000[1]), method='sum')



    channel_range_100=np.asarray([np.int64((lbda_range_100[0] - 7500) / 1.25)+1, np.int64((lbda_range_100[1] - 7500) / 1.25) + 1])
    channel_range_200=np.asarray([np.int64((lbda_range_200[0] - 7500) / 1.25)+1, np.int64((lbda_range_200[1] - 7500) / 1.25) + 1])
    channel_range_500=np.asarray([np.int64((lbda_range_500[0] - 7500) / 1.25)+1, np.int64((lbda_range_500[1] - 7500) / 1.25) + 1])
    channel_range_1000=np.asarray([np.int64((lbda_range_1000[0] - 7500) / 1.25)+1, np.int64((lbda_range_1000[1] - 7500) / 1.25) + 1])

    img_narrow_band_100.data, img_narrow_band_100.var = collapseCube(dataCube=cube.data, statCube=cube.var,minChannel=channel_range_100[0],maxChannel=channel_range_100[1])
    img_narrow_band_200.data, img_narrow_band_200.var = collapseCube(dataCube=cube.data, statCube=cube.var,minChannel=channel_range_200[0],maxChannel=channel_range_200[1])
    img_narrow_band_500.data, img_narrow_band_500.var = collapseCube(dataCube=cube.data, statCube=cube.var,minChannel=channel_range_500[0],maxChannel=channel_range_500[1])
    img_narrow_band_1000.data, img_narrow_band_1000.var = collapseCube(dataCube=cube.data, statCube=cube.var,minChannel=channel_range_1000[0],maxChannel=channel_range_1000[1])


    print(channel_range_100)
    print(channel_range_200)
    print(channel_range_500)
    print(channel_range_1000)

    flux_100, err_100 = quickApPhotmetryNoBg(imgData=img_narrow_band_100.data, imgStat=img_narrow_band_100.var, xObj=ra_three_accept_pix, yObj=dec_three_accept_pix, rObj=3.1)
    flux_200, err_200 = quickApPhotmetryNoBg(imgData=img_narrow_band_200.data, imgStat=img_narrow_band_200.var,xObj=ra_three_accept_pix, yObj=dec_three_accept_pix, rObj=3.1)
    flux_500, err_500 = quickApPhotmetryNoBg(imgData=img_narrow_band_500.data, imgStat=img_narrow_band_500.var,xObj=ra_three_accept_pix, yObj=dec_three_accept_pix, rObj=3.1)
    flux_1000, err_1000 = quickApPhotmetryNoBg(imgData=img_narrow_band_1000.data, imgStat=img_narrow_band_1000.var,xObj=ra_three_accept_pix, yObj=dec_three_accept_pix, rObj=3.1)

    flux_100=np.asarray(flux_100)
    err_100 = np.asarray(err_100)

    flux_200=np.asarray(flux_200)
    err_200 = np.asarray(err_200)

    flux_500=np.asarray(flux_500)
    err_500 = np.asarray(err_500)

    flux_1000=np.asarray(flux_1000)
    err_1000 = np.asarray(err_1000)

    ratio_100 = flux_100 / err_100
    ratio_200 = flux_200 / err_200
    ratio_500 = flux_500 / err_500
    ratio_1000 = flux_1000 / err_1000

    print(ratio_100.shape)
    print(ratio_200.shape)
    print(ratio_500.shape)
    print(ratio_1000.shape)

    mu_100,sigma_100 = norm.fit(ratio_100)
    mu_200, sigma_200 = norm.fit(ratio_200)
    mu_500, sigma_500 = norm.fit(ratio_500)
    mu_1000, sigma_1000 = norm.fit(ratio_1000)

    bin = np.linspace(-10, 10, 15)
    x = np.linspace(-10, 10, 500)
    fig, ax = plt.subplots(1, 4, figsize=(15, 15), tight_layout=True)


    ax[0].hist(ratio_100, bins=bin, density=True, alpha=0.6)
    p_1 = norm.pdf(x, mu_100, sigma_100)
    ax[0].plot(x, p_1, 'k')
    ax[0].set_title(f"100kms-1 \n $\mu$ = {np.round(mu_100, decimals=2)} $\sigma$ = {np.round(sigma_100, decimals=2)}")

    ax[1].hist(ratio_200, bins=bin, density=True, alpha=0.6)
    p_2 = norm.pdf(x, mu_200, sigma_200)
    ax[1].plot(x, p_2, 'k')
    ax[1].set_title(f"200kms-1 \n $\mu$ = {np.round(mu_200, decimals=2)} $\sigma$ = {np.round(sigma_200, decimals=2)}")

    ax[2].hist(ratio_500, bins=bin, density=True, alpha=0.6)
    p_3 = norm.pdf(x, mu_500, sigma_500)
    ax[2].plot(x, p_3, 'k')
    ax[2].set_title(f"500kms-1 \n $\mu$ = {np.round(mu_500, decimals=2)} $\sigma$ = {np.round(sigma_500, decimals=2)}")

    ax[3].hist(ratio_1000, bins=bin, density=True, alpha=0.6)
    p_4 = norm.pdf(x, mu_1000, sigma_1000)
    ax[3].plot(x, p_4, 'k')
    ax[3].set_title(
        f"1000kms-1 \n $\mu$ = {np.round(mu_1000, decimals=2)} $\sigma$ = {np.round(sigma_1000, decimals=2)}")

    plt.show()
    #plt.savefig(f"{histogram_and_fit_directory_path}/run_{z + 1}.png")
    #plt.close('all')










