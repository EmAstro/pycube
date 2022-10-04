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
        self.primary_extension = None
        self.data_extension = None
        self.error_extension = None
        self.error_type = None
        self.wavelength_cards = {}

    @property
    def data_extension(self):
        return self._data_extension

    @data_extension.setter
    def data_extension(self, data_extension):
        self._data_extension = data_extension

    @property
    def error_extension(self):
        return self._error_extension

    @error_extension.setter
    def error_extension(self, error_extension):
        self._error_extension = error_extension

    @property
    def primary_extension(self):
        return self._primary_extension

    @primary_extension.setter
    def primary_extension(self, primary_extension):
        self._primary_extension = primary_extension

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        self._error_type = error_type

    @property
    def wavelength_cards(self):
        return self._wavelength_cards

    @wavelength_cards.setter
    def wavelength_cards(self, wavelength_cards):
        self._wavelength_cards = wavelength_cards