import matplotlib.pyplot as plt
import torch
import os


def plot_target_vs_output(target, output, epoch, save_dir="plots"):
    # Only make the directory if it doesn't exist — safe even if it already does
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert tensors to CPU NumPy arrays
    target_np = target.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(target_np, label="Target", linewidth=2)
    plt.plot(output_np, label="Output", linestyle="--")
    plt.xlabel("Timestep")
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
