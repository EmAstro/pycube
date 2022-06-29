from pycube.instruments import instrument


class VLTMuse(instrument.Instrument):
    telescope = 'VLT'  # ToDo this will be made a class
    name = 'MUSE'
    data_extension = 'DATA'
    sigma_extension = 'STAT'

