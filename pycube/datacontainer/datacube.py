
from pycube.datacontainer import datacontainer


class DataCube(datacontainer.DataContainer):
    r"""This class is designed to dictat the behaviour of a datacube
    """
    def __init__(self, hdul=None, instrument=None, fits_file=None):
        super().__init__(hdul=hdul, instrument=instrument, fits_file=fits_file)

