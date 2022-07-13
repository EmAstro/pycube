class Instrument:
    """
    Class defining the behaviour of an ifu instrument in ``pycube``.

    Attributes:
    """

    telescope = None
    """
    Name of the telescope where the instrument is mounted
    """

    name = None
    """
    Name of the spectrograph
    """

    def __init__(self):
        self.data_extension = None
        self.sigma_extension = None

    @property
    def data_extension(self):
        return self._data_extension

    @data_extension.setter
    def data_extension(self, data_extension):
        self._data_extension = data_extension

    @property
    def sigma_extension(self):
        return self._sigma_extension

    @sigma_extension.setter
    def sigma_extension(self, sigma_extension):
        self._sigma_extension = sigma_extension

    @property
    def primary_extension(self):
        return self._primary_extension

    @primary_extension.setter
    def primary_extension(self, primary_extension):
        self._primary_extension = primary_extension
