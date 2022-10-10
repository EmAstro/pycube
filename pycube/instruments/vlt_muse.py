from pycube.instruments import instrument


class VLTMuse(instrument.Instrument):
    def __init__(self):
        self.primary_extension = 'PRIMARY'
        self.data_extension = 'DATA'
        self.error_extension = 'STAT'
        self.error_type = 'Variance'
        self.telescope = 'VLT'  # ToDo this will be made a class
        self.name = 'MUSE'
        self.wavelength_cards = {'CRVAL3': 'CRVAL3',
                                 'CRPIX3': 'CRPIX3',
                                 'CDELT3': 'CD3_3',
                                 'CUNIT3': 'CUNIT3'}
        self.update_units = False

