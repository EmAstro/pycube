import astropy.units as u

string_to_astropy_units = {
    'um': u.micron,
    'Angstrom': u.angstrom,
    '10**(-20)*erg/s/cm**2/Angstrom': 10**-20.*u.erg*u.s**-1*u.cm**-2*u.angstrom,
    'MJy/sr': 10**6.*u.Jy*u.sr**-1
}


def to_astropy_units(unit_string):
    """Converts units given as a string to an astropy quantity
    """
    return string_to_astropy_units[unit_string.strip()]

'''
def convert_flux_density_to_cgs(unit):
    if units 
'''