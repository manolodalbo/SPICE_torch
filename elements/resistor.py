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
            self.raw_R = torch.nn.Parameter(
                torch.tensor(resistance, dtype=torch.float32, device=self.device).log()
            )
        else:
            self.register_buffer(
                "raw_R",
                torch.tensor(resistance, dtype=torch.float32, device=self.device).log(),
            )
        self.n0 = n0
        self.n1 = n1
        self.timesteps = timesteps
        self.I_values = []
        self.track = track
        self.opt = train
        self.lr = 0.1

    @property
    def R(self):
        return torch.exp(self.raw_R)

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

    def __str__(self):
        return (
            f"Resistor '{self.name}':\n"
            f"  Nodes: n0 = {self.n0}, n1 = {self.n1}\n"
            f"  Resistance: {self.R.item():.4f} Ω\n"
            f"  Trainable: {self.opt}\n"
            f"  Learning Rate: {self.lr}\n"
            f"  Tracking Enabled: {self.track}\n"
            f"  Timesteps: {self.timesteps}\n"
            f"  Device: {self.device}"
        )
