import torch


class Cap(torch.nn.Module):
    def __init__(
        self,
        name,
        capacitance,
        n0,
        n1,
        timestep,
        timesteps,
        device,
        track: bool = False,
        train: bool = False,
    ):
        super().__init__()
        self.name = name
        if train:
            self.raw_C = torch.nn.Parameter(
                torch.tensor(capacitance, dtype=torch.float32, device=device).log()
            )
        else:
            self.register_buffer(
                "raw_C",
                torch.tensor(capacitance, dtype=torch.float32, device=device).log(),
            )
        self.n0 = n0
        self.n1 = n1
        self.prev = torch.tensor(0.0, device=device)
        self.timestep = timestep
        self.timesteps = timesteps
        self.I_values = []
        self.track = track
        self.opt = train
        self.lr = 0.1

    @property
    def C(self):
        return torch.exp(self.raw_C)

    def I(self, V0, V1):
        i = self.C * ((V1 - V0) - self.prev) / self.timestep
        self.prev = V1 - V0
        self.I_values.append(i)
        return i

    def G(self, A, b):
        A[self.n0, self.n0] += self.C / self.timestep
        A[self.n0, self.n1] += -self.C / self.timestep
        b[self.n0] += self.C * self.prev / self.timestep
        A[self.n1, self.n1] += self.C / self.timestep
        A[self.n1, self.n0] += -self.C / self.timestep
        b[self.n1] -= self.C * self.prev / self.timestep
        return A, b

    def get_lr(self):
        return self.lr

    def __str__(self):
        return f"capacitor: {self.name} with C={self.C} and self.prev={self.prev}"
