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
