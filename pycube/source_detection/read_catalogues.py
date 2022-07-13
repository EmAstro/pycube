from astropy.io import fits


def read_lsd_output(fits_file):
    """

    """
    table = fits.open(fits_file)
