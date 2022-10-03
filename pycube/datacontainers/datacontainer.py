import numpy as np

from astropy.io import fits
from pycube.ancillary import checks
from pycube import msgs

from pycube.instruments import vlt_muse

__all__ = ['DataContainer']


class DataContainer:
    r"""Base class to dictate the general behavior of a data container

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
            msgs.work('Loading datacube...')
            self.hdul = fits.open(fits_file)
            msgs.info('Datacube loaded')
        else:
            raise ValueError('Error in reading in {}'.format(fits_file))
        if self.instrument is None:
            # try to get the instrument from the primary header
            if 'INSTRUME' in self.hdul[0].header:
                if self.hdul[0].header['INSTRUME'] == 'MUSE':
                    self.instrument = vlt_muse
                    msgs.info('Instrument set to vlt_muse')
                else:
                    msgs.warning('Instrument {} not initialized'.format(self.hdul[0].header['INSTRUME']))
            else:
                msgs.warning('Instrument not defined')

    def get_data_hdu(self, data_extension=None):
        """Get the HDU for the data extension

        """
        if data_extension is not None:
            return self.hdul[data_extension]
        elif self.instrument is not None:
            return self.hdul[self.instrument.data_extension]
        else:
            msgs.warning('data_extension needs to be specified')
            return None

    def get_data(self, data_extension=None, copy=True):
        """Get the data for the data extension

        """
        if copy:
            return np.copy(self.get_data_hdu(data_extension=data_extension).data)
        else:
            return self.get_data_hdu(data_extension=data_extension).data

    def get_data_header(self, data_extension=None):
        """Get the header for the data extension

        """
        return self.get_data_hdu(data_extension=data_extension).header

    def get_error(self, error_extension=None, copy=True):
        """Get the data for the error extension

        """
        if copy:
            return np.copy(self.get_data_hdu(error_extension=error_extension).data)
        else:
            return self.get_data_hdu(error_extension=error_extension).data

    def get_error_hdu(self, error_extension=None):
        """Get the HDU for the data extension

        """
        if error_extension is not None:
            return self.hdul[error_extension]
        elif self.instrument is not None:
            return self.hdul[self.instrument.error_extension]
        else:
            msgs.warning('error_extension needs to be specified')
            return None

    def get_error_header(self, error_extension=None):
        """Get the header for the data extension

        """
        return self.get_data_hdu(error_extension=error_extension).header

    def copy(self):
        """Returns a shallow copy

        """
        return DataContainer(hdul=self.hdul.copy(), fits_file=self.fits_file, instrument=self.instrument)

