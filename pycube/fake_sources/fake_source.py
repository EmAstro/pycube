from astropy.modeling import models


class FakeSource:
    """"

    """
    def __init__(self, location=(None, None, None), spatial_shape=None, spectral_shape=None):
        self.location = location
        self.spatial_shape = spatial_shape
        self.spectral_shape = spectral_shape


class GaussianSource(FakeSource):
    r"""This class is designed
    """
    def __init__(self, location=None, spectral_shape=None):
        super().__init__(location=location, spatial_shape=models.Gaussian2D(), spectral_shape=spectral_shape)
