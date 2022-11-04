import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from pycube.ancillary import checks
from pycube.ancillary import units
from pycube import msgs


from pycube.instruments import vlt_muse
from pycube.instruments import jwst_nirspec

__all__ = ['DataContainer']

DEFAULT_FLUX_DENSITY_UNITS = 10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom**-1
DEFAULT_WAVELENGTH_UNITS = 1.*u.angstrom
ERROR_TYPES = {'None': {'Descriptor': 'No error type defined'},
               'SIGMA': {'Descriptor': 'Error extension contains 1 sigma'},
               'VAR': {'Descriptor': 'Error extension contains variance'},
               'IVAR': {'Descriptor': 'Error extension contains the inverse variance'}}


class DataContainer:
    r"""Base class to dictate the general behavior of a data container

    This class is oriented to work with units of 10**-20.*u.erg*u.s**-1*u.cm**-2*Ang.**-1 and wavelength in Ang.
    By default the code will try to convert the units in your cube and update the values in the fits header (this is
    now implemented only for NIRSpec).

    Attributes:

    """
    def __init__(self, hdul=None, instrument=None, fits_file=None):
        self.hdul = hdul
        self.instrument = instrument
        self.fits_file = fits_file

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def hdul(self):
        return self._hdul

    @hdul.setter
    def hdul(self, hdul):
        self._hdul = hdul

    @property
    def fits_file(self):
        return self._fits_file

    @fits_file.setter
    def fits_file(self, fits_file):
        if fits_file is None:
            self._fits_file = None
        elif checks.fits_file_is_valid(fits_file):
            self._fits_file = fits_file
            msgs.work('Loading data...')
            self.hdul = fits.open(fits_file, output_verify='silentfix')
            msgs.info('Data loaded')
        else:
            raise ValueError('Error in reading in {}'.format(fits_file))
        if self.instrument is None:
            # try to get the instrument from the primary header
            if 'INSTRUME' in self.hdul[0].header:
                # ToDo: this selection should work using a dictionary instead of being hardcoded
                if self.hdul[0].header['INSTRUME'] == 'MUSE':
                    self.instrument = vlt_muse
                    msgs.info('Instrument set to vlt_muse')
                if self.hdul[0].header['INSTRUME'] == 'NIRSPEC':
                    self.instrument = jwst_nirspec
                    msgs.info('Instrument set to jwst_nirspec')
                else:
                    msgs.warning('Instrument {} not initialized'.format(self.hdul[0].header['INSTRUME']))
            else:
                msgs.warning('Instrument not defined')

    def get_data_hdu(self, extension=None):
        """Get the HDU for the data extension

        """
        if extension is None:
            extension = self.instrument.data_extension
        return self._get_hdu(extension=extension)

    def get_data(self, extension=None, copy=True):
        """Get the data for the data extension

        """
        if copy:
            return np.copy(self.get_data_hdu(extension=extension).data)
        else:
            return self.get_data_hdu(extension=extension).data

    def get_data_header(self, header_card=None, extension=None):
        """Get the header for the data extension

        If an header card is entered, the code will return the corresponding value in the header
        """
        if header_card is None:
            return self.get_data_hdu(extension=extension).header
        else:
            return self.get_data_hdu(extension=extension).header[header_card]

    def get_error(self, extension=None, copy=True):
        """Get the data for the error extension

        """
        if copy:
            return np.copy(self.get_error_hdu(extension=extension).data)
        else:
            return self.get_error_hdu(extension=extension).data

    def get_error_hdu(self, extension=None):
        """Get the HDU for the data extension

        """
        if extension is None:
            extension = self.instrument.error_extension
        return self._get_hdu(extension=extension)

    def get_error_header(self, header_card=None, extension=None):
        """Get the header for the data extension

        If an header card is entered, the code will return the corresponding value in the header
        """
        if header_card is None:
            return self.get_error_hdu(extension=extension).header
        else:
            return self.get_error_hdu(extension=extension).header[header_card]

    def _get_hdu(self, extension=None):
        """Get the HDU given an extension

        """
        if extension is not None:
            return self.hdul[extension]
        else:
            msgs.warning('extension needs to be specified')
            return None

    def get_pixel_area(self, extension=None):
        """Extract the pixel scale from the header

        """
        wcs = WCS(self.get_data_header(extension=extension))
        return wcs.proj_plane_pixel_area()

    def get_updated_data_header(self, header_card=None, header_value=None, extension=None):
        """Update header value

        """
        if extension is None:
            extension = self.instrument.data_extension
        self.hdul[extension].header[header_card] = header_value

    def copy(self):
        """Returns a shallow copy

        """
        return DataContainer(hdul=self.hdul.copy(), fits_file=self.fits_file, instrument=self.instrument)

    def save(self, output_fits_file, overwrite=False):
        """Save the fits file

        Args:
            output_fits_file (str): name of the new output fits file
            overwrite (bool, optional): if `True`, overwrite the `output_fits_file` it exists.
        """
        msgs.work('Saving data..')
        self.hdul.writeto(output_fits_file, output_verify='silentfix', overwrite=overwrite, checksum=True)
        msgs.info('File {} written'.format(output_fits_file))