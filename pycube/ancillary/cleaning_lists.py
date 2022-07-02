""" Module that performs some lists massaging to work well with arg.parser
"""


import numpy as np
from astropy.table import MaskedColumn
import astropy.units as u

from pycube import msgs
from pycube.ancillary import checks


def make_list_of_fits_files(args_input, verify_fits=False):
    r"""Cleaning an input list of fits files
    Args:
        args_input (list): input list of fits files that will be checked (usually is coming from `parse_arguments()` in
            a script)
        verify_fits (bool): if set to `True`, it will verify that the fits file is complaint to the FITS standard
    Returns:
        list: list containing all the valid fits files given in input
    """
    list_of_fits_files = []
    if not isinstance(args_input, list):
        args_input_files: list = [args_input]
    else:
        args_input_files: list = args_input
    for args_input_file in args_input_files:
        if checks.fits_file_is_valid(args_input_file, overwrite=False, verify_fits=verify_fits):
            list_of_fits_files.append(args_input_file)
        else:
            msgs.warning('{} excluded because not a valid fits file'.format(args_input_file))
    if len(list_of_fits_files) == 0:
        msgs.error('No valid fits files present')
    return list_of_fits_files


def from_element_to_list(element, element_type=str):
    r"""Given an element it returns a list containing the element
    It also checks all the elements in the list have the same type defined by `element_type`
    Args:
        element (any): element that will be put in the list
        element_type (any): type of the element that should be contained in the list
    Returns:
        list: list containing `element`
    """
    if element is None:
        return None
    elif isinstance(element, list):
        for element_in_list in element:
            assert isinstance(element_in_list, element_type), r'{} must be a {}'.format(element_in_list, element_type)
        return element
    elif isinstance(element, np.ndarray):
        element_list: list = element.tolist()
        for element_in_list in element_list:
            assert isinstance(element_in_list, element_type), r'{} must be a {}'.format(element_in_list, element_type)
        return element_list
    elif isinstance(element, MaskedColumn):
        element_list = element.data.data.tolist()
        for element_in_list in element_list:
            assert isinstance(element_in_list, element_type), r'{} must be a {}'.format(element_in_list, element_type)
        return element_list
    elif isinstance(element, element_type):
        return [element]
    else:
        msgs.error('Not valid type for: {}'.format(element))
    return


def from_element_to_list_of_quantities(element, unit=None):
    r"""Convert an input into a list of `astropy.units.Quantity`
    Args:
        element (int, float, np.ndarray, Quantity object, list): element that will be put in the list
        unit (UnitBase instance): An object that represents the unit to be associated with the input value
    Returns:
        list: list of quantities in the format `element`*`unit`
    """
    assert isinstance(unit, u.UnitBase), r'{} not a valid astropy units'.format(unit)
    if isinstance(element, int):
        return [float(element)*unit]
    elif isinstance(element, float):
        return [element*unit]
    elif isinstance(element, np.ndarray) and not isinstance(element, u.Quantity):
        element_list_clean = []
        element_list: list = np.copy(element).tolist()
        for element_in_list in element_list:
            element_list_clean.append(element_in_list*unit)
        return element_list_clean
    elif isinstance(element, u.Quantity):
        element_list_clean = []
        element_converted = np.copy(element).to(unit)
        for element_in_list in np.nditer(element_converted):
            print(element_in_list)
            element_list_clean.append(element_in_list*unit)
        return element_list_clean
    elif isinstance(element, list):
        element_list_clean = []
        for element_in_list in element:
            if isinstance(element_in_list, u.Quantity):
                element_list_clean.append(element_in_list.to(unit))
            else:
                element_list_clean.append(element_in_list*unit)
        return element_list_clean
    else:
        msgs.error('The input cannot be converted into a list of quantities')
        return
