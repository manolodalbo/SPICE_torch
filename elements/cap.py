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
            self.C = torch.nn.Parameter(
                torch.tensor(capacitance, dtype=torch.float32, device=device)
            )
        else:
            self.register_buffer(
                "C", torch.tensor(capacitance, dtype=torch.float32, device=device)
            )
        self.n0 = n0
        self.n1 = n1
        self.prev = torch.tensor(0.0, device=device)
        self.timestep = timestep
        self.timesteps = timesteps
        self.I_values = []
        self.track = track
        self.opt = train
        self.lr = 0.01

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
