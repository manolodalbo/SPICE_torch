import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simulated target behavior: R = 20Ω
true_resistance = 20.0
v_start, v_end = 0.0, 5.0
timesteps = 1000
voltages = torch.linspace(v_start, v_end, timesteps)
target_currents = voltages / true_resistance  # I = V / R


# Learnable resistor model
class ResistorModel(nn.Module):
    def __init__(self, initial_R=5.0):
        super().__init__()
        self.raw_R = nn.Parameter(torch.tensor(initial_R, dtype=torch.float32).log())

    @property
    def R(self):
        return torch.exp(self.raw_R)

    def forward(self, V):
        return V / self.R


# Instantiate model and optimizer
model = ResistorModel(initial_R=5.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1)
criterion = nn.MSELoss()

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    predicted_currents = model(voltages)
    loss = criterion(predicted_currents, target_currents)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 999:
        print(
            f"Epoch {epoch:3d} | R = {model.R.item():.4f} Ω | Loss = {loss.item():.6f}"
        )

# Final resistance
print(f"\nFinal learned resistance: {model.R.item():.4f} Ω")

# Plotting
with torch.no_grad():
    predicted = model(voltages)
    plt.plot(voltages, target_currents, label="Target (20Ω)")
    plt.plot(voltages, predicted, label=f"Learned ({model.R.item():.2f}Ω)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title("Resistor I-V Curve Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
