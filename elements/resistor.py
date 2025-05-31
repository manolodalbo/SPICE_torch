import torch


class Resistor(torch.nn.Module):
    def __init__(
        self, name, resistance, n0, n1, track: bool = False, train: bool = False
    ):
        super().__init__()
        self.name = name
        if train:
            self.R = torch.nn.Parameter(torch.tensor(resistance, dtype=torch.float32))
        else:
            self.register_buffer("R", torch.tensor(resistance, dtype=torch.float32))
        self.n0 = n0
        self.n1 = n1
        self.I_values = []
        self.track = track
        self.opt = train

    def I(self, V1, V0):
        I = (V1 - V0) / self.R
        self.I_values.append(I)
        return I

    def G(self):
        return ((self.n0, 1 / self.R), (self.n1, 1 / self.R))
