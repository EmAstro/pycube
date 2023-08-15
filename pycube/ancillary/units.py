"""
Module to deal with different units in a fits file

.. include:: ../../docs/source/include/links.rst
"""

import astropy.units as u
import numpy as np

# ToDo this needs to be updated to be less dependent on the way the units are written.
STRING_TO_ASTROPY_QUANTITY = {
    'um':
        {'quantity': 1.*u.micron,
         'type': 'wavelength'},
    'Angstrom':
        {'quantity': 1.*u.angstrom,
         'type': 'wavelength'},
    '10**(-20)*erg/s/cm**2/Angstrom':
        {'quantity': 10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom**-1,
         'type': 'f_lambda'},
    'MJy/sr':
        {'quantity': 1.*u.MJy*u.sr**-1,
         'type': 'f_nu_over_sr'}
}


def to_astropy_quantity(quantity_string):
    """Converts quantity given as a string to an astropy `astropy.units.Quantity`_

    The behavior is dictated by the dictionary: `units.STRING_TO_ASTROPY_QUANTITY`.

    Args:
        quantity_string (str): string to be mapped in an `astropy.units.Quantity`_

    Returns:
        `astropy.units.Quantity`_: resulting quantity
    """
    if quantity_string.strip() not in STRING_TO_ASTROPY_QUANTITY.keys():
        raise KeyError('Quantity {} not mapped. '.format(quantity_string) +
                       'Possible values are: {}'.format(list(STRING_TO_ASTROPY_QUANTITY.keys())))
    return STRING_TO_ASTROPY_QUANTITY[quantity_string.strip()]['quantity']


def _to_astropy_quantity_and_type(quantity_string):
    """Converts quantity given as a string to an astropy `astropy.units.Quantity`_ and its type.

    The behavior is dictated by the dictionary: `units.STRING_TO_ASTROPY_QUANTITY`. The type information is used
    internally by the code to perform the correct conversion.

    Args:
        quantity_string (str): string to be mapped in an `astropy.units.Quantity`_

    Returns:
        tuple: resulting `astropy.units.Quantity`_ and internal type as a str
    """
    if quantity_string.strip() not in STRING_TO_ASTROPY_QUANTITY.keys():
        raise KeyError('Quantity {} not mapped. '.format(quantity_string) +
                       'Possible values are: {}'.format(list(STRING_TO_ASTROPY_QUANTITY.keys())))
    return STRING_TO_ASTROPY_QUANTITY[quantity_string.strip()]['quantity'], \
           STRING_TO_ASTROPY_QUANTITY[quantity_string.strip()]['type']


def to_string_quantity(quantity_astropy):
    """Converts `astropy.units.Quantity`_ names into a string

    The behavior is dictated by the dictionary: `units.STRING_TO_ASTROPY_QUANTITY`.

    Args:
        quantity_astropy (`astropy.units.Quantity`_): quantity to be mapped into a string

    Returns:
        str: resulting string
    """
    quantity_string = None
    for quantity_string_map, quantity_astropy_map in STRING_TO_ASTROPY_QUANTITY.items():
        if quantity_astropy == quantity_astropy_map['quantity']:
            quantity_string = quantity_string_map
    if quantity_string is None:
        raise KeyError('Unit {} not mapped. '.format(quantity_astropy) +
                       'Possible values are: {}'.format(list(STRING_TO_ASTROPY_QUANTITY.values())))
    return quantity_string


def update_wavelength_units_in_header(hdul, CRVAL3, CDELT3, CUNIT3, to_quantity=1.*u.angstrom):
    """Update the header information to convert the data to a specific wavelength unit

    .. warning::
        The code assumes that the wavelengths are linearly spaced (and not in log).

    Args:
        hdul (`astropy.io.fits.HDUList`_): `astropy.io.fits.HDUList`_ in which the header information will be updated

    Results:
        hdul (`astropy.io.fits.HDUList`_): `astropy.io.fits.HDUList`_ in which the header information will be updated
        CRVAL3 (str): base value header card that will be updated
        CDELT3 (str): delta wavelength header card that will be updated
        CUNIT3 (str): wavelength unit header card that will be updated
        to_quantity (`astropy.units.Quantity`_): quantity in which the wavelength information will be converted into

    Returns:
        `astropy.io.fits.HDUList`_: updated `astropy.io.fits.HDUList`_
    """
    for hdu in hdul:
        if CUNIT3 in hdu.header:
            if hdu.header[CUNIT3] != to_string_quantity(to_quantity):
                hdu.header[CRVAL3] = (hdu.header[CRVAL3]*to_astropy_quantity(hdu.header[CUNIT3])).to(to_quantity).value
                hdu.header[CDELT3] = (hdu.header[CDELT3]*to_astropy_quantity(hdu.header[CUNIT3])).to(to_quantity).value
                hdu.header[CUNIT3] = to_string_quantity(to_quantity)
    return hdul


def update_spectral_flux_units_in_header(hdul, wavelength_vector, pixel_area,
                                         data_extension, error_extension, error_type,
                                         to_quantity=10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom**-1):
    """Update the header information to convert the data to a specific flux density unit

    .. warning::
        Currently this works only on the 'DATA' and 'STAT' extension. If your cube contains any other extension,
        the unit of these will not be updated.

    Args:
        hdul (`astropy.io.fits.HDUList`_): `astropy.io.fits.HDUList`_ in which the header information will be updated
        wavelength_vector (`astropy.units.Quantity`_): Wavelength vector used for flux density correction
        pixel_area  (`astropy.units.Quantity`_): Area of a pixel used for surface brightness correction
        data_extension (str): extension of the data information
        error_extension (str): extension of the error information
        error_type (str): take into account if the error information is provided as sigma, variance, or inverse of the
            variance.
        to_quantity (`astropy.units.Quantity`_): quantity in which the wavelength information will be converted into

    Returns:
        `astropy.io.fits.HDUList`_: updated `astropy.io.fits.HDUList`_
    """
    for extension in [data_extension, error_extension]:
        if 'BUNIT' in hdul[extension].header:
            if hdul[extension].header['BUNIT'] != to_string_quantity(to_quantity):
                astropy_quantity, astropy_type = _to_astropy_quantity_and_type(hdul[extension].header['BUNIT'])
                if astropy_type == 'f_nu_over_sr':
                    # To avoid issues with the conversion of units. Everything moves to cgs and then is translated
                    # back into the requested units.
                    astropy_quantity = astropy_quantity * pixel_area.to(u.sr)
                    scale_factor = astropy_quantity.to(to_quantity, equivalencies=u.spectral_density(
                        wavelength_vector)).value
                    if extension == data_extension:
                        hdul[extension].header['BUNIT'] = to_string_quantity(to_quantity)
                        hdul[extension].data = hdul[extension].data * np.expand_dims(scale_factor, axis=(1, 2))
                    if extension == error_extension:
                        if error_type == 'SIGMA':
                            hdul[extension].header['BUNIT'] = to_string_quantity(to_quantity)
                            hdul[extension].data = hdul[extension].data * np.expand_dims(scale_factor, axis=(1, 2))
                        else:
                            raise NotImplementedError(
                                'The requested error type {} is not implemented yet'.format(error_type))
                else:
                    raise NotImplementedError('The requested unit type {} is not implemented yet'.format(astropy_type))
    return hdul
