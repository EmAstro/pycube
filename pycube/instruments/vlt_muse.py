from pycube.instruments import instrument


class VLTMuse(instrument.Instrument):
    def __init__(self):
        self.primary_extension = 'PRIMARY'
        self.data_extension = 'DATA'
        self.sigma_extension = 'STAT'
        self.telescope = 'VLT'  # ToDo this will be made a class
        self.name = 'MUSE'



