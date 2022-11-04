from pycube.instruments import instrument


class JWSTNIRSpec(instrument.Instrument):
    def __init__(self):
        self.primary_extension = 'PRIMARY'
        self.data_extension = 'SCI'
        self.error_extension = 'ERR'
        self.error_type = 'SIGMA'
        self.telescope = 'JWST'  # ToDo this will be made a class
        self.name = 'NIRSpec'
        self.wavelength_cards = {'CRVAL3': 'CRVAL3',
                                 'CRPIX3': 'CRPIX3',
                                 'CDELT3': 'CDELT3',
                                 'CUNIT3': 'CUNIT3'}
        self.update_units = True

