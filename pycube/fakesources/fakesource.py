from astropy.modeling import models


class FakeSource:
    """"

    """
    def __init__(self, x_spatial=None, y_spatial=None, z_spectral=None, spatial_shape=None, spectral_shape=None):
        self.x_spatial = x_spatial
        self.y_spatial = y_spatial
        self.z_spectral = z_spectral
        self.spatial_shape = spatial_shape
        self.spectral_shape = spectral_shape


class GaussianSource(FakeSource):
    r"""This class is designed
    """
    def __init__(self, x_spatial=None, y_spatial=None, z_spectral=None, spectral_shape=None,
                 **kwargs):
        super().__init__(x_spatial=x_spatial, y_spatial=y_spatial, z_spectral=z_spectral,
                         spatial_shape=models.Gaussian2D(x_mean=x_spatial, y_mean=y_spatial,
                                                         **kwargs),
                         spectral_shape=models.Gaussian1D(mean=z_spectral))


