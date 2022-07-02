r"""
pycube_collapse
===============
Script to collapse a datacontainer along the spectra axis

.. topic:: Inputs:

    - **input_cube** - Input data_cube
    - **wavelength_min** - Minimum wavelength
    - **wavelength_max** - Maximum wavelength
"""

import argparse

from pycube import __version__

EXAMPLES = str(r"""EXAMPLES:""" + """\n""" + """\n""" +
               r"""pycube_collapse""" + """\n""" +
               r""" """)


def parser(options=None):
    parser = argparse.ArgumentParser(
        description=r"""Take datacubes and collapse it along the spectra axis.  """ +
                    """\n""" + """\n""" +
                    r"""In case wavelengths limits are provide, these will be considered as limit for """ +
                    r"""to create the image. The output will also contain the error and the background""" +
                    r"""if present in the extensions. """ +
                    """\n""" + """\n""" +
                    r"""This uses pycube version {:s}""".format(__version__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES)

    parser.add_argument("fits_files", nargs="+", type=str,
                        help=r"Input datacontainer")
    parser.add_argument("-wl_min", "--wavelength_min", nargs="+", type=float, default=None,
                        help=r"Lower limit to collapse the cube")
    parser.add_argument("-wl_max", "--wavelength_max", action="store_true", default=False,
                        help=r"Upper limit to collapse the cube")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    return parser.parse_args()


def main(args):

    from pycube import msgs
    from pycube.ancillary import cleaning_lists

    # Cleaning input lists
    fits_files = cleaning_lists.make_list_of_fits_files(args.fits_files)
    # Cleaning input values
    # ToDo units should be added
    wavelength_min = cleaning_lists.from_element_to_list(args.wavelength_min, element_type=float)[0]
    wavelength_max = cleaning_lists.from_element_to_list(args.wavelength_max, element_type=float)[0]

    msgs.start()
    for fits_file in fits_files:
        msgs.work('Collapsing cube: {}'.format(fits_file))

