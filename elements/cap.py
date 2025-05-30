import torch


class Cap(torch.nn.Module):
    def __init__(
        self,
        name,
        capacitance,
        n0,
        n1,
        timestep,
        track: bool = False,
        train: bool = False,
    ):
        super().__init__()
        self.name = name
        if train:
            self.C = torch.nn.Parameter(torch.tensor(capacitance, dtype=torch.float32))
        else:
            self.register_buffer("C", torch.tensor(capacitance, dtype=torch.float32))
        self.n0 = n0
        self.n1 = n1
        self.prev = 0
        self.timestep = timestep
        self.I_values = []
        self.track = track

    def I(self, V0, V1):
        self.prev = V1 - V0
        i = self.C * ((V1 - V0) - self.prev) / self.timestep
        self.I_values.append(i)
        return i

    def G(self):
        return (
            (self.n0, self.C / self.timestep),
            (self.n1, -self.C / self.timestep),
            (-1, -self.C + self.prev),
        )
