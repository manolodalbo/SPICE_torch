class Resistor:
    def __init__(self, name, resistance, show_IV: bool):
        self.name = name
        self.R = resistance
        self.IV = show_IV
        self.I_values = []

    def I(self, V1, V0):
        I = (V1 - V0) / self.R
        self.I_values.append(I)
        return I

    def G(self):
        return ((1 / self.R), (-1 / self.R))
