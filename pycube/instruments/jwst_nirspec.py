from pycube.instruments import instrument


class JWSTNIRSpec(instrument.Instrument):
    def __init__(self):
        self.primary_extension = 'PRIMARY'
        self.data_extension = 'SCI'
        self.error_extension = 'ERR'
        self.error_type = 'Sigma'
        self.telescope = 'JWST'  # ToDo this will be made a class
        self.name = 'NIRSpec'

