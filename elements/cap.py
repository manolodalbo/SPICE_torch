class Cap:
    def __init__(self, name, capacitance, n0, n1, timestep):
        self.name = name
        self.C = capacitance
        self.n0 = n0
        self.n1 = n1
        self.prev = 0
        self.timestep = timestep
        self.current = []

    def I(self, V0, V1):
        self.prev = V1 - V0
        i = self.C * ((V1 - V0) - self.prev) / self.timestep
        self.current.append(i)
        return i

    def G(self):
        return (
            (self.n0, self.C / self.timestep),
            (self.n1, -self.C / self.timestep),
            (-1, -self.C + self.prev),
        )
