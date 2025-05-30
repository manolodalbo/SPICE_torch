class Resistor:
    def __init__(self, name, resistance, n0, n1, track: bool = False):
        self.name = name
        self.R = resistance
        self.n0 = n0
        self.n1 = n1
        self.I_values = []
        self.track = track

    def I(self, V1, V0):
        I = (V1 - V0) / self.R
        self.I_values.append(I)
        return I

    def G(self):
        return ((self.n0, 1 / self.R), (self.n1, -1 / self.R))
