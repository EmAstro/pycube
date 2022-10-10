import numpy as np
from pycube import msgs
from pycube.ancillary import units
from pycube.datacontainers import datacontainer
from astropy import units as u


class DataCube(datacontainer.DataContainer):
    r"""This class is designed to dictate the behaviour of a datacube
    """
    def __init__(self, hdul=None, instrument=None, fits_file=None):
        super().__init__(hdul=hdul, instrument=instrument, fits_file=fits_file)

    def get_wavelength_vector(self, with_units=True):
        """Returns a vector with the same size of the spectral axis but with the wavelength information in it

        This is calculated from the 'CRVAL3', 'CDELT3', and 'CUNIT3'

        Args:
            with_units (bool): if set to `True` the code will try to get the unit information from the
                header card 'CUNIT3' and attach it to the result

        Returns:
            np.array, quantity: array of wavelengths
        """
        wavelength = self.get_data_header(header_card=
                                          self.instrument.wavelength_cards['CRVAL3']) + \
                     self.get_channel_vector()*self.get_data_header(header_card=
                                                                    self.instrument.wavelength_cards['CDELT3'])
        if with_units:
            wavelength = wavelength * \
                         units.to_astropy_units(self.get_data_header(header_card=
                                                                     self.instrument.wavelength_cards['CUNIT3']))
        return wavelength

    def get_wavelength_min_max(self, with_units=True):
        """Returns the spectral range covered by the datacube

        Args:
            with_units (bool): if set to `True` the code will try to get the unit information from the
                header card 'CUNIT3' and attach it to the result

        Returns:
            tuple: min_wavelength and max_wavelength as floats. If the `with_units` is set to `True` these will be
                quantities
        """
        _wavelength = self.get_wavelength_vector(with_units=with_units)
        return np.nanmin(_wavelength), np.nanmax(_wavelength)

    def get_channel_vector(self):
        """Returns a vector with the same size of the spectral axis

        Returns:
            np.array: starting from 0 to the total channel in the cube spectral direction
        """
        z_spectral, _, _ = self.get_data_size()
        return np.arange(z_spectral)

    def get_channel_given_wavelength(self, wavelength_value):
        """Given a wavelength, it returns the corresponding channel

        Args:
            wavelength_value (float, quantity): wavelength corresponding to the channel

        Returns
            int: closest spectral channel corresponding to the wavelength
        """
        if isinstance(wavelength_value, u.quantity.Quantity):
            with_units = True
        else:
            with_units = False
        wavelength = self.get_wavelength_vector(with_units=with_units)
        if with_units:
            wavelength_value.to_value(wavelength.unit)
        wavelength_min, wavelength_max = self.get_wavelength_min_max(with_units=with_units)
        if (wavelength_value<wavelength_min) or (wavelength_value>wavelength_max):
            msgs.warning(r'{:.4f} outside the wavelength range {:.4f} < lambda < {:.4f}'.format(wavelength_value,
                                                                                                wavelength_min,
                                                                                                wavelength_max))
            return None
        difference_in_wavelength = np.absolute(wavelength - wavelength_value)
        return difference_in_wavelength.argmin()

    def get_data_size(self):
        """Return the size of a cube in z, y, and x

        This is just running numpy.shape over the datacube
        """
        z_spectral, y_spatial, x_spatial = np.shape(self.get_data())
        return z_spectral, y_spatial, x_spatial

    def collapse(self, min_wavelength=None, max_wavelength=None, mask_wavelength=None, to_flux=True):
        """Collapse the cube along the spectral axis given a range of values.

        If 'mask_wavelength' is specified, the channel marked as 'True' will be excluded from the collapsing

        Args:
            min_wavelength (float, quantity): Minimum wavelength from where to collapse the cube
            max_wavelength (float, quantity): Maximum wavelength to where the cube will be collapsed
            mask_wavelength (np.array): Boolean array with the same length of the spectral axis of the cube.
                Only channels set to `False` will be used for the collapsing
            to_flux (bool): if 'True' the result is multiplied by the wavelength bin to have the result in
                10**-20.*u.erg*u.s**-1*u.cm**-2 (this is currently hard coded, but it may change in the future). If
                'False' it will be a simple sum over the selected channels (thus the units are likely to be wrong).

        Returns:
            tuple: data and errors derived from the collapsing process. These are image.Image objects.
        """
        if to_flux:
            scale_factor = 1.
            msgs.work(r'Converting the units to fluxes')
        else:
            scale_factor = 1.

        # test for quantities
        if isinstance(min_wavelength, u.quantity.Quantity) and isinstance(max_wavelength, u.quantity.Quantity):
            with_units = True
        elif isinstance(min_wavelength, u.quantity.Quantity) and not isinstance(max_wavelength, u.quantity.Quantity):
            raise TypeError(r'min_wavelength type is different from max_wavelength type')
        elif not isinstance(min_wavelength, u.quantity.Quantity) and isinstance(max_wavelength, u.quantity.Quantity):
            raise TypeError(r'min_wavelength type is different from max_wavelength type')
        else:
            with_units = False
        _min_wavelength_cube, _max_wavelength_cube = self.get_wavelength_min_max(with_units=with_units)
        if min_wavelength is None:
            min_wavelength = _min_wavelength_cube
        if max_wavelength is None:
            max_wavelength = _max_wavelength_cube
        _min_channel_cube = self.get_channel_given_wavelength(min_wavelength)
        _max_channel_cube = self.get_channel_given_wavelength(max_wavelength)



'''
def collapse_cube(datacube,
                  min_lambda=None,
                  max_lambda=None,
                  mask_z=None):
    """ Given a 3D data/stat cube .FITS file, this function collapses along the z-axis given a range of values.
    If mask_z is specified, the function will remove all channels masked as 1.

    Parameters
    ----------
    datacube : np.array
        3D data cube
    min_lambda : int, optional
        Minimum wavelength to collapse file
    max_lambda : int, optional
        Maximum wavelength to collapse file
    mask_z : np.array, optional
        range of z-axis values to mask when collapsing

    Returns
    -------
    np.array
        Condensed 2D array of 3D file.
    """

    datacopy = np.copy(datacube)
    z_max, y_max, x_max = np.shape(datacopy)

    # Checks and resets if outside boundaries of z
    if max_lambda is None or max_lambda > z_max:
        max_lambda = z_max
        print("collapse_cube : Exceeded / unspecified wavelength in data cube. "
              "Max value is set to {}".format(int(z_max)))
    if min_lambda is None or min_lambda < 0:
        min_lambda = 0
        print("collapse_cube : Invalid / unspecified minimum wavelength. Min value is set to 0")
    if mask_z is not None:
        datacopy[mask_z, :, :] = np.nan

    # Sums values between specifications, ignoring NaNs
    col_cube = np.nansum(datacopy[min_lambda:max_lambda, :, :], axis=0)
    del datacopy
    del z_max, y_max, x_max
    return col_cube
'''