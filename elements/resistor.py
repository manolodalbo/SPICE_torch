import torch


class Resistor(torch.nn.Module):
    def __init__(
        self,
        name,
        resistance,
        n0,
        n1,
        timesteps,
        device,
        track: bool = False,
        train: bool = False,
    ):
        super().__init__()
        self.device = device
        self.name = name
        if train:
            self.R = torch.nn.Parameter(
                torch.tensor(resistance, dtype=torch.float32, device=self.device)
            )
        else:
            self.register_buffer(
                "R", torch.tensor(resistance, dtype=torch.float32, device=self.device)
            )
        self.n0 = n0
        self.n1 = n1
        self.timesteps = timesteps
        self.I_values = []
        self.track = track
        self.opt = train
        self.lr = 0.1

    def I(self, V1, V0):
        I = (V1 - V0) / self.R
        self.I_values.append(I)
        return I

    def G(self, A, b):
        g = 1 / self.R
        A[self.n0, self.n0] += g
        A[self.n0, self.n1] -= g
        A[self.n1, self.n1] += g
        A[self.n1, self.n0] -= g
        return A, b

    def get_lr(self):
        return self.lr
