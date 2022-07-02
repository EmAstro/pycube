from astropy.io import fits
from pycube.ancillary import checks

__all__ = ['DataContainer']


class DataContainer:
    r"""Base class to dictat the general behaviour of a datacontainer

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
        if checks.fits_file_is_valid(fits_file):
            self._fits_file = fits_file
        else:
            raise ValueError('Error in reading in {}'.format(fits_file))
        self.hdul = fits.open(fits_file)










