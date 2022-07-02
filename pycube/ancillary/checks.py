r"""Module that performs some useful and basic checks and transformations
"""

# import sys
import numpy as np
import shutil
import urllib
import os.path

from astropy.io import fits

from pycube import msgs


def fits_file_is_valid(fits_file, verify_fits=False, overwrite=False):
    r"""Check if a file exists and has a valid extension

    The option `verify_fits` checks the header of the fits file using `astropy.io.fits.verify`
    Args:
        fits_file (str): fits file you would like to check
        verify_fits (bool): if set to `True`, it will verify that the fits file is complaint to the FITS standard.
        overwrite (bool): if `True`, overwrite the input fits file with the header corrections from `verify_fits`
    Returns:
        bool: `True` if exists `False` and warning raised if not.
    """
    is_fits = True
    # Checks if it is a string
    assert isinstance(fits_file, str), 'input file needs to be a string'
    # Check for ending
    if '.fit' not in fits_file.lower():
        msgs.warning('File: {} does not have a fits extension`'.format(fits_file))
        is_fits = False
    # Check for existence
    if not os.path.exists(fits_file):
        msgs.warning('File: {} does not exists'.format(fits_file))
        is_fits = False
    # Check for compliance with FITS standard
    if verify_fits:
        if overwrite:
            hdul = fits.open(fits_file, mode='update', checksum=False)
            if not check_checksums(hdul):
                is_fits = False
            hdul.flush(output_verify='fix+warn', verbose=True)
            hdul.writeto(fits_file, checksum=True, overwrite=True)
            msgs.info('File checked and rewritten')
        else:
            hdul = fits.open(fits_file, mode='readonly', checksum=True)
            if not check_checksums(hdul):
                is_fits = False
            hdul.verify('fix+warn')
        hdul.close()
    else:
        if overwrite:
            msgs.error('The option overwrite works only if verify_fits = True')
    return is_fits


def check_checksums(hdul):
    r"""Test if the `datasum` and `checksum` keywords in a `HDUList` are present and up-to-date
    Args:
        hdul (`HDUList`_): list of `astropy`_ HDUs to be checked
    Returns:
        bool: `True` all the HDUs in the input `HDUList`_ have the correct `datasum` and `checksum`
    """
    is_good_checksum = True
    for hdu in hdul:
        checks_for_checksum = hdu.verify_checksum()
        checks_for_datasum = hdu.verify_datasum()
        if checks_for_checksum == 0:
            msgs.warning('Checksum not valid')
            is_good_checksum = False
        if checks_for_checksum == 2:
            msgs.warning('Checksum not present')
            is_good_checksum = False
        if checks_for_datasum == 0:
            msgs.warning('Datasum not valid')
            is_good_checksum = False
        if checks_for_datasum == 2:
            msgs.warning('Datasum not present')
            is_good_checksum = False
    return is_good_checksum
