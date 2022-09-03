from astropy.modeling import models


class FakeSource:
    """"

    """
    def __init__(self, x_spatial=None, y_spatial=None, z_spectral=None,
                 spatial_shape=None, spectral_shape=None):
        self.x_spatial = x_spatial
        self.y_spatial = y_spatial
        self.z_spectral = z_spectral
        self.spatial_shape = spatial_shape
        self.spectral_shape = spectral_shape

        def evaluate():
            return


class GaussianSource(FakeSource):
    r"""This class is designed
    """
    def __init__(self, x_spatial=None, y_spatial=None, z_spectral=None,
                 sigma_spatial=None, sigma_spectral=None):
        super().__init__(x_spatial=x_spatial, y_spatial=y_spatial, z_spectral=z_spectral,
                         spatial_shape=models.Gaussian2D(x_mean=x_spatial, y_mean=y_spatial,
                                                         x_stddev=sigma_spatial, y_stddev=sigma_spatial,
                                                         amplitude=1., theta=0.),
                         spectral_shape=models.Gaussian1D(mean=z_spectral, stddev=sigma_spectral,
                                                          amplitude=1.))


