import numpy as np
from pycube.ancillary import units
from pycube.datacontainers import datacontainer


class DataCube(datacontainer.DataContainer):
    r"""This class is designed to dictate the behaviour of a datacube
    """
    def __init__(self, hdul=None, instrument=None, fits_file=None):
        super().__init__(hdul=hdul, instrument=instrument, fits_file=fits_file)

    def get_wavelength_vector(self, with_units=True):
        """Returns a vector with the same size of the spectral axis but with the wavelength information in it

        This is calculated from the 'CRVAL3' and 'CDELT3'
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

    def get_channel_vector(self):
        """Returns a vector with the same size of the spectral axis
        """
        z_spectral, _, _ = self.get_data_size()
        return np.arange(z_spectral)

    def get_data_size(self):
        """Return the size of a cube in z, y, and x

        This is just running numpy.shape over the datacube
        """
        z_spectral, y_spatial, x_spatial = np.shape(self.get_data())
        return z_spectral, y_spatial, x_spatial

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