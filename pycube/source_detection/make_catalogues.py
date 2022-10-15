import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import astropy.units as u
from mpdaf.obj import Cube, Spectrum,WaveCoord
from astropy.io import fits
from astropy import wcs
from photutils import aperture_photometry,CircularAperture,CircularAnnulus
from scipy import ndimage
import warnings

def detect(zap,cat):
    np.seterr(divide='ignore',invalid='ignore') #ignores true divide error
    warnings.filterwarnings('ignore')           #ignore warnings
    f = fits.open(zap)
    hdul = fits.open(cat)  # LSDcat CATALOGUE
    cube = Cube(zap)
    w1 = wcs.WCS(f[1].header)
    hdr = f[1].header  # to get data header info
    r = 0.0002222

    print("Creating Main Directory in Desktop")
    directory = "Detection_pycube"
    desktop_path = os.path.expanduser('~') + '/Desktop'
    dir_path = os.path.join(desktop_path, directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(
            dir_path)  # Removes all the subdirectories! (use for cleaning up the old directory and create a new on)
        os.makedirs(dir_path)
    print(f"DETECTION DIRECTORY CREATED\n")

    print("")
    print("CREATING WHITE IMAGE")
    white = cube.sum(axis=0)
    w = fits.open(zap)
    w[1].data = np.nansum(w[1].data, axis=0)
    w.writeto(f'{dir_path}/white.fits')
    w.close()
    print("WHITE IMAGE CREATED")
    print("")




    print("CREATING LSDcat DIRECTORY")
    lsdcat_directory_path = os.path.join(dir_path,"LSDcat")
    if not os.path.exists(lsdcat_directory_path):
        os.mkdir(lsdcat_directory_path)
    else:
        shutil.rmtree(lsdcat_directory_path)
        os.mkdir(lsdcat_directory_path)
    print("LSDcat DIRECTORY CREATED\n")

    data = hdul[1].data

    x = []
    y = []
    z = []
    id = []
    for i in range(data.shape[0]):
        id.append(data[i][1])
        x.append(data[i][2])
        y.append(data[i][3])
        z.append(data[i][4])
    id = np.asarray(id)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    #Covertring the "All Detections" voxels to world coordinates. This process takes some time to convert
    ra = []
    dec = []
    wave = []
    for i in range(x.shape[0]):
        t = w1.all_pix2world([[x[i], y[i], z[i]]], 0)
        ra.append(t[0][0])
        dec.append(t[0][1])
        wave.append(t[0][2])
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    wave = np.asarray(wave)




    # Coverting "Unique Individual Objects" voxels to world coordinates. This process takes some time to convert
    count_arr = np.bincount(id)
    print(count_arr)
    print(len(count_arr))
    print(f"length of id = {len(id)}")
    print(f"length of x_check = {len(x)}")
    x_average = []
    y_average = []
    z_average = []
    n = []

    #take i as len(bin)
    j = 0
    for i in range(1, len(count_arr)):
        x_average.append(np.average(x[j:j + count_arr[i]]))
        y_average.append(np.average(y[j:j + count_arr[i]]))
        n.append(i)
        j = j + count_arr[i]
    x_average = np.asarray(x_average)
    y_average = np.asarray(y_average)

    ra_avg = []
    dec_avg = []
    wave_avg = []

    for i in range(x_average.shape[0]):
        t = w1.all_pix2world([[x_average[i], y_average[i], z[i]]], 0)
        ra_avg.append(t[0][0])
        dec_avg.append(t[0][1])

    ra_avg = np.asarray(ra_avg)
    dec_avg = np.asarray(dec_avg)

    j = 0
    c = 0
    wave_save = np.zeros(shape=(len(count_arr) - 1, np.amax(count_arr)))
    for i in range(1, len(count_arr)):
        for k in range(count_arr[i]):
            wave_save[j][k] = wave[c]
            c = c + 1
        j = j + 1





    print("Removing Edges")
    # Edges Removed
    x_edges_removed = np.asarray([20, hdr['NAXIS1'] - 20])
    y_edges_removed = np.asarray([20, hdr['NAXIS2'] - 20])
    ra_edges_removed = []
    dec_edges_removed = []
    n_edges_removed=[]


    for i in range(x_average.shape[0]):
        if x_edges_removed[0] < x_average[i] < x_edges_removed[1] and y_edges_removed[0] < y_average[i] < y_edges_removed[1]:
            t2 = w1.all_pix2world([[x_average[i], y_average[i], z[i]]], 0)
            ra_edges_removed.append(t2[0][0])
            dec_edges_removed.append(t2[0][1])
            n_edges_removed.append(n[i])




    ra_edges_removed = np.asarray(ra_edges_removed)
    dec_edges_removed = np.asarray(dec_edges_removed)
    n_edges_removed=np.asarray(n_edges_removed)



    wave_edges_removed = np.empty(shape=(len(ra_edges_removed), np.amax(count_arr)))





    print(f"Shape of original ra : {len(ra_avg)}")
    print(f"Shape of cropped ra : {len(ra_edges_removed)}")

    wave_edges_removed = np.empty(shape=(len(n_edges_removed), np.amax(count_arr)))
    l=0
    n_check=[]
    for i in range(len(ra_edges_removed)):
        for j in range(len(ra_avg)):
            if ra_edges_removed[i] == ra_avg[j]:
                n_check.append(j+1)
                for k in range(len(wave_save[j][:])):
                    wave_edges_removed[l][k]=wave_save[j][k]
                l=l+1





    #CREATING LSDcat CATALOGUE
    print("CREATING LSDcat CATALOGUE AND SAVING IN LSDcat DIRECTORY")
    name_of_catalogue = os.path.join(lsdcat_directory_path, "LSDcat_Catalog.txt")
    file_1 = open(name_of_catalogue, "w+")
    file_1.write("ID, ra, dec, wavelengths\n")
    for i in range(len(ra_edges_removed)):
        file_1.write(f"{i+1},    {n_edges_removed[i]},   {ra_edges_removed[i]}, {dec_edges_removed[i]}, {np.trim_zeros(wave_edges_removed[i])}\n")
    file_1.close()
    print("CATALOGUE CREATED\n")




    #CREATING DS9 REGION FILE
    print("CREATING DS9 REGION FILE")
    name_of_region_file = os.path.join(lsdcat_directory_path, "LSDcat_region_file.reg")
    file_2 = open(name_of_region_file, "w+")
    file_2.write("#Region file format: DS9 version 4.1 "
               "\nglobal color=blue dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 "
               "\nfk5\n")
    for i in range(len(ra_edges_removed)):
        file_2.write(f"circle({ra_edges_removed[i]},{dec_edges_removed[i]},{r})\n")
    file_2.close()
    print("FILE CREATED\n")

    ###########################################################################################################

    n_sky_line = []

    ra_sky_line = []
    dec_sky_line = []

    ra_possible_source = []
    dec_possible_source = []
    wave_possible_source = []

    for i in range(len(ra_edges_removed)):
        m = 0
        t = np.trim_zeros(wave_edges_removed[i])
        t = np.asarray(t)
        for j in range(len(t)):
            if 7.75125e-7 < t[j] < 9.200e-7:
                m = m + 1
            else:
                continue

        if m == 0:
            n_sky_line.append(n_edges_removed[i])
            ra_sky_line.append(ra_edges_removed[i])
            dec_sky_line.append(dec_edges_removed[i])

        elif m != 0:
            ra_possible_source.append(ra_edges_removed[i])
            dec_possible_source.append(dec_edges_removed[i])
            wave_possible_source.append(t)

    ra_sky_line = np.asarray(ra_sky_line)
    dec_sky_line = np.asarray(dec_sky_line)

    ra_possible_source = np.asarray(ra_possible_source)
    dec_possible_source = np.asarray(dec_possible_source)
    wave_possible_source = np.asarray(wave_possible_source,dtype=object)

    print("")
    print(f"NUMBER OF POSSIBLE SOURCES: {len(ra_possible_source)}")
    print("")

    # CREATING DS9 REGION FILE FILE FOR SPURIOUS SOURCES
    print("CREATING DS9 REGION FILE FOR SPURIOUS SOURCES")
    name_of_region_file = os.path.join(lsdcat_directory_path, "LSDcat_spurious_sources_region_file.reg")
    file_3 = open(name_of_region_file, "w+")
    file_3.write("#Region file format: DS9 version 4.1 "
                 "\nglobal color=red dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 "
                 "\nfk5\n")
    for i in range(len(ra_sky_line)):
        file_3.write(f"circle({ra_sky_line[i]},{dec_sky_line[i]},{0.0002778})\n")
    file_3.close()
    print("FILE CREATED\n")



    ###########################################################################################################
    #datafile='/home/sai/Desktop/P183p05_DATACUBE_ZAP.fits'
    #cube = Cube(datafile)


    step = cube.get_step()
    wave_step = step[0]
    cube.data = cube.data * wave_step



    delta_lbda_100 = int((1e2 / 3e5) * 9001.)
    delta_lbda_500 = int((5e2 / 3e5) * 9001.)
    delta_lbda_1000 = int((1e3 / 3e5) * 9001.)
    lbda_range_100 = np.asarray([9001. - delta_lbda_100, 9001. + delta_lbda_100])
    lbda_range_500 = np.asarray([9001. - delta_lbda_500, 9001. + delta_lbda_500])
    lbda_range_1000 = np.asarray([9001. - delta_lbda_1000, 9001. + delta_lbda_1000])

    print("")
    print("")
    print("CREATING IMAGES")
    lsdcat_images_directory_path = os.path.join(lsdcat_directory_path, "IMAGES")
    if not os.path.exists(lsdcat_images_directory_path):
        os.mkdir(lsdcat_images_directory_path)
    else:
        shutil.rmtree(lsdcat_images_directory_path)
        os.mkdir(lsdcat_images_directory_path)

    lsdcat_original_spectra_directory_path = os.path.join(lsdcat_directory_path, "ORIGINAL_SPECTRA")
    if not os.path.exists(lsdcat_original_spectra_directory_path):
        os.mkdir(lsdcat_original_spectra_directory_path)
    else:
        shutil.rmtree(lsdcat_original_spectra_directory_path)
        os.mkdir(lsdcat_original_spectra_directory_path)


    flux_1=[]
    flux_2=[]
    flux_3=[]

    a_mean=[]
    a_median=[]
    b_mean=[]
    b_median=[]
    b_sum=[]
    n=[]
    c=0
    k=0
    ra_potential_sources=[]
    dec_potential_sources = []
    wave_potential_sources =[]



    wavelengths=[]
    wave_start = f['DATA'].header['CRVAL3']
    for i in range(f['DATA'].header['NAXIS3']):
        wavelengths.append(wave_start)
        wave_start+=wave_step
    wavelengths=np.asarray(wavelengths)
    wavelengths=1e-10 * wavelengths

    #Image Creation len(ra_possible_source)
    for i in range(len(ra_possible_source)):
        print(f"RUN {i+1}")

        ra_img = ra_possible_source[i]
        dec_img = dec_possible_source[i]
        d = cube.wcs.sky2pix((dec_img, ra_img))
        ra_img_pix = d[0][1]
        dec_img_pix = d[0][0]

        def spectra_extraction(x, y):
            #CUBE = file  # Input datacube
            xObj = x  # x-position of the object in pixel
            yObj = y  # y-position of the object in pixel
            rObj = 5.  # extraction radius in pixel
            rIbg = 10.  # internal radius for the background
            rObg = 20.  # external radius for the background
            outSpec = 'spec.txt'  # output filename
            isiraf = True


            # Check for format
            if type(xObj) is list:
                xObj = float(xObj[0])
            if type(yObj) is list:
                yObj = float(yObj[0])
            if type(rObj) is list:
                rObj = float(rObj[0])
            if type(rIbg) is list:
                rIbg = float(rIbg[0])
            if type(rObg) is list:
                rObg = float(rObg[0])
            if type(outSpec) is list:
                outSpec = str(outSpec[0])

            if isiraf:
                print(" ")
                print('Moving from IRAF to python pixel system')
                print(" ")
                xObj = xObj + 1.
                yObj = yObj + 1.

            print(" ")
            print("Working on the cube")
            print(" ")
            print("Opening fits files")
            #fitsCube = fits.open(CUBE)
            # fitsCube.info()
            dataCube = f['DATA'].data
            statCube = f['STAT'].data
            zMax, yMax, xMax = dataCube.shape

            posObj = [xObj, yObj]
            circObj = CircularAperture(posObj, r=rObj)
            annuObj = CircularAnnulus(posObj, r_in=rIbg, r_out=rObg)

            specApPhot = []
            specVarApPhot = []
            specChannelPhot = []

            print(" ")
            print("Extracting spectrum")

            dataCubeChannel = np.arange(0, zMax, 1)
            for channel in dataCubeChannel:
                apPhot = aperture_photometry(dataCube[channel, :, :], circObj)
                bgPhot = aperture_photometry(dataCube[channel, :, :], annuObj)
                varApPhot = aperture_photometry(statCube[channel, :, :], circObj)
                specApPhot.append(apPhot['aperture_sum'][0] - (bgPhot['aperture_sum'][0] * circObj.area / annuObj.area))
                specVarApPhot.append(varApPhot['aperture_sum'][0])
                specChannelPhot.append(float(channel))

            specWavePhot = f['DATA'].header['CRVAL3'] + (
                    np.array(specChannelPhot) * f['DATA'].header['CD3_3'])

            # os.system("rm -fr " + outSpec)
            # specTxt = open(outSpec, 'w')
            # np.savetxt(specTxt,
            #          np.c_[specChannelPhot, specWavePhot, specApPhot, specVarApPhot])
            # specTxt.close()

            print(" ")





            return specApPhot,np.power(specVarApPhot, 0.5)

        flux, err = spectra_extraction(ra_img_pix, dec_img_pix)
        flux = np.asarray(flux)
        err = np.asarray(err)

        for y in range(len(wavelengths)):
            if wavelengths[y] == 8998.75e-10:
                flux_1.append(flux[i])
                flux_2.append(flux[i+1])
                flux_3.append(flux[i+2])
            else:
                continue

        a=[]
        b=[]
        for j in range(0,1402):
            a.append(flux[j])
        for j in range(1402,1481):
            b.append(flux[j])

        a=np.asarray(a)
        b=np.asarray(b)


        print(f"b_mean = {np.nanmean(b)}")
        print(f"b_sum = {np.nansum(b)}")



        if  np.nanmean(b)>-500 and np.nansum(b)>-20000:
            ra_potential_sources.append(ra_img)
            dec_potential_sources.append(dec_img)
            wave_potential_sources.append(wave_possible_source[i])
            a_mean.append(np.nanmean(a))
            a_median.append(np.nanmedian(a))
            b_mean.append(np.nanmean(b))
            b_median.append(np.nanmedian(b))
            b_sum.append(np.nansum(b))
            c = c + 1
            n.append(c)

            sub_cube_test = cube.subcube((dec_img, ra_img), size=13)
            img_broad_band_test = sub_cube_test.get_image((7500, 8703.75), method='sum')
            img_narrow_band_100_test = sub_cube_test.get_image((lbda_range_100[0], lbda_range_100[1]), method='sum')
            img_narrow_band_500_test = sub_cube_test.get_image((lbda_range_500[0], lbda_range_500[1]), method='sum')
            img_narrow_band_1000_test = sub_cube_test.get_image((lbda_range_1000[0], lbda_range_1000[1]), method='sum')

            s2n_broad_band_test = img_broad_band_test.copy()
            s2n_narrow_band_100_test = img_narrow_band_100_test.copy()
            s2n_narrow_band_500_test = img_narrow_band_500_test.copy()
            s2n_narrow_band_1000_test = img_narrow_band_1000_test.copy()

            smooth_s2n_broad_band_test = img_broad_band_test.copy()
            smooth_s2n_narrow_band_100_test = img_narrow_band_100_test.copy()
            smooth_s2n_narrow_band_500_test = img_narrow_band_500_test.copy()
            smooth_s2n_narrow_band_1000_test = img_narrow_band_1000_test.copy()

            def s2n_smooth(data,var):
                sSmooth = 1.
                truncate = 5.
                dataSigma = (sSmooth, sSmooth)
                statSigma = (sSmooth / np.sqrt(2.), sSmooth / np.sqrt(2.))

                dataTmp = np.nan_to_num(data, copy='True')
                statTmp = np.copy(var)
                statTmp[np.isnan(var)] = np.nanmax(var)

                # smooth cubes

                sDataImage = ndimage.filters.gaussian_filter(dataTmp, dataSigma,
                                                             truncate=truncate)
                sStatImage = ndimage.filters.gaussian_filter(statTmp, statSigma,
                                                             truncate=truncate)

                del dataSigma
                del statSigma
                return dataTmp/np.sqrt(statTmp),sDataImage/np.sqrt(sStatImage)

            s2n_broad_band_test.data, smooth_s2n_broad_band_test.data=s2n_smooth(s2n_broad_band_test.data,s2n_broad_band_test.var)
            s2n_narrow_band_100_test.data, smooth_s2n_narrow_band_100_test.data = s2n_smooth(s2n_narrow_band_100_test.data,s2n_narrow_band_100_test.var)
            s2n_narrow_band_500_test.data, smooth_s2n_narrow_band_500_test.data = s2n_smooth(s2n_narrow_band_500_test.data,s2n_narrow_band_500_test.var)
            s2n_narrow_band_1000_test.data, smooth_s2n_narrow_band_1000_test.data = s2n_smooth(s2n_narrow_band_1000_test.data,s2n_narrow_band_1000_test.var)


            wave_spec = WaveCoord(cdelt=f['DATA'].header['CD3_3'], crval=f['DATA'].header['CRVAL3'], cunit=u.angstrom)
            spec = Spectrum(data=flux, var=err, wave=wave_spec, unit=u.Unit(str(f['DATA'].header['BUNIT'])))
            spec_var=Spectrum(data=err, wave=wave_spec, unit=u.Unit(str(f['DATA'].header['BUNIT'])))


            print(f"{ra_possible_source[i]} : {dec_possible_source[i]}")
            fig,((ax17),(ax18))=plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
            circleObj = plt.Circle((ra_img_pix, dec_img_pix), 5, color='DarkBlue', fill=False)
            plt.gcf().gca().add_artist(circleObj)
            spec.plot(ax=ax17)
            spec_var.plot(ax=ax17)
            white.plot(ax=ax18, zscale=True)


            plt.suptitle(f"{c} :  ra = {ra_possible_source[i]}    dec = {dec_possible_source[i]}")
            plt.savefig(f"{lsdcat_original_spectra_directory_path}/{c}.png", bbox_inches='tight')
            plt.close('all')
            # plt.show()

            fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(15, 15), tight_layout=True)
            plt.suptitle(f"{c} :  ra = {ra_possible_source[i]}    dec = {dec_possible_source[i]}")
            img_broad_band_test.plot(ax=ax1, zscale=True, colorbar='v', scale='linear', title="Broad Band")
            img_narrow_band_100_test.plot(ax=ax2, zscale=True, colorbar='v', scale='linear', title="100kms")
            img_narrow_band_500_test.plot(ax=ax3, zscale=True, colorbar='v', scale='linear', title="500kms")
            img_narrow_band_1000_test.plot(ax=ax4, zscale=True, colorbar='v', scale='linear', title="1000kms")

            s2n_broad_band_test.plot(ax=ax5,zscale=True,scale='linear',colorbar='v',title="s2n")
            s2n_narrow_band_100_test.plot(ax=ax6, zscale=True, scale='linear', colorbar='v', title="s2n")
            s2n_narrow_band_500_test.plot(ax=ax7, zscale=True, scale='linear', colorbar='v', title="s2n")
            s2n_narrow_band_1000_test.plot(ax=ax8, zscale=True, scale='linear', colorbar='v', title="s2n")

            smooth_s2n_broad_band_test.plot(ax=ax9,zscale=True,scale='linear',colorbar='v',title="smooth")
            smooth_s2n_narrow_band_100_test.plot(ax=ax10,zscale=True,scale='linear',colorbar='v',title="smooth")
            smooth_s2n_narrow_band_500_test.plot(ax=ax11, zscale=True, scale='linear', colorbar='v', title="smooth")
            smooth_s2n_narrow_band_1000_test.plot(ax=ax12, zscale=True, scale='linear', colorbar='v', title="smooth")


            spec.plot(ax=ax13, lmin=7500, lmax=8703.75)
            spec_var.plot(ax=ax13, lmin=7500, lmax=8703.75)
            spec.plot(ax=ax14, lmin=lbda_range_100[0], lmax=lbda_range_100[1])
            spec_var.plot(ax=ax14, lmin=lbda_range_100[0], lmax=lbda_range_100[1])
            spec.plot(ax=ax15, lmin=lbda_range_500[0], lmax=lbda_range_500[1])
            spec_var.plot(ax=ax15, lmin=lbda_range_500[0], lmax=lbda_range_500[1])
            spec.plot(ax=ax16, lmin=lbda_range_1000[0], lmax=lbda_range_1000[1])
            spec_var.plot(ax=ax16, lmin=lbda_range_1000[0], lmax=lbda_range_1000[1])
            plt.savefig(f"{lsdcat_images_directory_path}/{c}.png")
            plt.close('all')
            # plt.show()

        else:
            pass

        #################################################################################################################







    print("IMAGES CREATED")
    print("")
    print(f"NUMBER OF POTENTIAL SOURCES : {len(ra_potential_sources)}")
    print("")
    # CREATING DS9 REGION FILE FOR POTENTIAL TRUE SOURCES
    print("CREATING DS9 REGION FILE FOR POTENTIAL TRUE SOURCES")
    name_of_region_file = os.path.join(lsdcat_directory_path, "LSDcat_potential_sources_region_file.reg")
    file_4 = open(name_of_region_file, "w+")
    file_4.write("#Region file format: DS9 version 4.1 "
                 "\nglobal color=green dashlist=8 3 width=1 font='helvetica 10 normal roman' select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 "
                 "\nfk5\n")
    for i in range(len(ra_potential_sources)):
        file_4.write(f"circle({ra_potential_sources[i]},{dec_potential_sources[i]},{0.0002778})\n")
    file_4.close()
    print("FILE CREATED\n")


    #CREATING LSDcat CATALOGUE FOR POTENTIAL TRUE SOURCES
    print("CREATING LSDcat CATALOGUE FOR POTENTIAL TRUE SOURCES")
    name_of_catalogue = os.path.join(lsdcat_directory_path, "LSDcat_potential_sources.txt")
    file_5 = open(name_of_catalogue, "w+")
    file_5.write("ID, ra, dec, wavelengths\n")
    for i in range(len(ra_potential_sources)):
        file_5.write(f"{i+1}, {ra_potential_sources[i]}, {dec_potential_sources[i]}, {wave_potential_sources[i]}\n")
    file_5.close()
    print("CATALOGUE CREATED\n")


    n=np.asarray(n)
    name_of_catalogue = os.path.join(lsdcat_directory_path, "LSDcat_flux_mean_median.txt")
    file_6 = open(name_of_catalogue, "w+")

    for i in range(len(a_mean)):
        file_6.write(f"{n[i]} : a_mean = {a_mean[i]}, a_median = {a_median[i]}, b_mean = {b_mean[i]}, b_median = {b_median[i]}, b sum = {b_sum[i]} \n\n")
    file_6.close()
    print("means and medians created\n")

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")




    wave_count = np.zeros(f['DATA'].header['NAXIS3'])
    wave_count_2=[]
    for i in range(len(ra_edges_removed)):
        for j in range(len(np.trim_zeros(wave_edges_removed[i]))):
            for k in range(len(wavelengths)):

                if wavelengths[k] == wave_edges_removed[i][j]:
                    wave_count[k]+=1
                    wave_count_2.append(wave_edges_removed[i][j])
                else:
                    continue

    print("HI")
    for i in range(len(wave_count)):
        print(f"{wavelengths[i]} : {wave_count[i]}")

    print("CREATING wavelength count catalogue")
    name_of_catalogue = os.path.join(lsdcat_directory_path, "wave_count.txt")
    file_7 = open(name_of_catalogue, "w+")
    file_7.write("Wavelength : Count\n")
    for i in range(len(wave_count)):
        if wave_count[i] !=0.0:
            file_7.write(f"{wavelengths[i]} : {wave_count[i]}\n")
        else:
            continue
    file_7.close()
    print("CATALOGUE CREATED\n")

    print(flux_1)
    print(flux_2)
    print(flux_3)

    print(f"{np.amin(flux_1)}, {np.amax(flux_1)}")
    print(f"{np.amin(flux_2)}, {np.amax(flux_2)}")
    print(f"{np.amin(flux_3)}, {np.amax(flux_3)}")

    return wavelengths, wave_count_2, flux_1, flux_2, flux_3