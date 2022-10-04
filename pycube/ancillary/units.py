import astropy.units as u

string_to_astropy_units = {
    'um': u.micron,
    'Angstrom': u.angstrom
}


def to_astropy_units(unit_string):
    """Converts units given as a string to an astropy quantity
    """
    return string_to_astropy_units[unit_string.strip()]