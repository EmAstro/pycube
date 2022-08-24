from pycube.instruments import instrument


class VLTMuse(instrument.Instrument):
    def __init__(self):
        self.primary_extension = 'PRIMARY'
        self.data_extension = 'DATA'
        self.error_extension = 'STAT'
        self.error_type = 'Variance'
        self.telescope = 'VLT'  # ToDo this will be made a class
        self.name = 'MUSE'


