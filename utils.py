import matplotlib.pyplot as plt
import torch
import os


def plot_target_vs_output(target, output, epoch, hyst, source, save_dir="plots"):
    timesteps = len(target)
    v_i = source.start
    v_f = source.end
    half = timesteps // 2 + 1

    if hyst:
        up = torch.linspace(v_i, v_f, half)
        down = torch.linspace(v_f, v_i, half)[1:]
        voltages = torch.cat([up, down])
    else:
        voltages = torch.linspace(v_i, v_f, timesteps)

    # Ensure voltages matches the target/output length
    voltages = voltages[:timesteps]  # in case of off-by-one in hysteresis mode

    # Make directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert tensors to CPU NumPy arrays
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    voltages_np = voltages.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(voltages_np, target_np, label="Target", linewidth=2)
    plt.plot(voltages_np, output_np, label="Output", linestyle="--")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Value")
    plt.title(f"Target vs Output (Epoch {epoch})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    filename = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(filename)
    plt.close()


def softplus_inverse(x):
    # Numerically stable inverse of softplus: log(exp(x) - 1)
    return torch.log(torch.expm1(x))
