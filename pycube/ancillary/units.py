"""
Module to deal with different units in a fits file

.. include:: ../../docs/source/include/links.rst
"""

import astropy.units as u
from astropy.wcs import WCS
from astropy import constants as const

string_to_astropy_quantity = {
    'um': 1.*u.micron,
    'Angstrom': 1.*u.angstrom,
    '10**(-20)*erg/s/cm**2/Angstrom': 10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom**-1,
    'MJy/sr': 1.*u.MJy*u.sr**-1
}


def to_astropy_quantity(quantity_string):
    """Converts quantity given as a string to an astropy quantity
    """
    if quantity_string.strip() not in string_to_astropy_quantity.keys():
        raise KeyError('Quantity {} not mapped. '.format(quantity_string) +
                       'Possible values are: {}'.format(list(string_to_astropy_quantity.keys())))
    return string_to_astropy_quantity[quantity_string.strip()]


def to_string_quantity(quantity_astropy):
    """Converts astropy quantity names into a string
    """
    quantity_string = None
    for quantity_string_map, quantity_astropy_map in string_to_astropy_quantity.items():
        if quantity_astropy == quantity_astropy_map:
            quantity_string = quantity_string_map
    if quantity_string is None:
        raise KeyError('Unit {} not mapped. '.format(quantity_astropy) +
                       'Possible values are: {}'.format(list(string_to_astropy_quantity.values())))
    return quantity_string


def update_wavelength_units_in_header(hdul, CRVAL3, CDELT3, CUNIT3, to_quantity=1.*u.angstrom):
    """Update the header information to convert the data to a specific wavelength unit
    """
    for hdu in hdul:
        if CUNIT3 in hdu.header:
            if hdu.header[CUNIT3] != to_string_quantity(to_quantity):
                hdu.header[CRVAL3] = (hdu.header[CRVAL3]*to_astropy_quantity(hdu.header[CUNIT3])).to(to_quantity).value
                hdu.header[CDELT3] = (hdu.header[CDELT3]*to_astropy_quantity(hdu.header[CUNIT3])).to(to_quantity).value
                hdu.header[CUNIT3] = to_string_quantity(to_quantity)
    return hdul


def update_spectral_flux_units_in_header(hdul, wavelength_vector, data_extension, error_extension, error_type,
                                         to_quantity=10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom**-1):
    """Update the header information to convert the data to a specific flux density unit

    Currently this works only on the 'DATA' and 'STAT' extension. If your cube contains any other extension, the
    unit of these will not be updated.

    Args:
        hdul (`astropy.io.fits.HDUList`_):
        wavelength_vector (`astropy.units.Quantity`_):
        data_extension (str):
        error_extension (str):
        error_type (str):
        to_quantity (`astropy.units.Quantity`_):

    Returns:
        `astropy.io.fits.HDUList`_:
    """
    for hdu in hdul:
        if 'BUNIT' in hdu.header:
            wcs = WCS(hdu.header)
            # pixel_scale = wcs.proj_plane_pixel_area()
    return hdul

'''
def convert_flux_density_to_cgs(unit):
    if units 
'''