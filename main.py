import argparse
from elements.resistor import Resistor
from elements.cap import Cap
from parse import parse_source, parse_target
import torch
import matplotlib.pyplot as plt


def sim(source, elements, number_of_nodes, timesteps, sweep_time, device):
    v_i = source.start
    v_f = source.end
    voltages = torch.linspace(v_i, v_f, int(timesteps), device=device)
    for i, voltage in enumerate(voltages):
        A = torch.zeros((number_of_nodes, number_of_nodes), device=device)
        b = torch.zeros((number_of_nodes, 1), device=device)
        tracking = None
        for element in elements:
            if isinstance(element, Resistor):
                if i == 0:
                    element.I_values = []
                A, b = element.G(A, b)
                if element.track:
                    if tracking is None:
                        tracking = element
                    else:
                        print(
                            f"Warning: Multiple tracking elements found. Using {tracking.name} for tracking."
                        )
            elif isinstance(element, Cap):
                if i == 0:
                    element.I_values = []
                    element.prev = torch.tensor(0.0, device=device)
                A, b = element.G(A, b)
                if element.track:
                    if tracking is None:
                        tracking = element
                    else:
                        print(
                            f"Warning: Multiple tracking elements found. Using {tracking.name} for tracking."
                        )
            else:
                print(f"Unknown element type: {type(element)}")
        A[0, :] = 0
        A[0, 0] = 1
        A[source.n1, :] = 0
        A[source.n1, source.n1] = 1
        A[source.n1, source.n0] = -1
        b[source.n1] = voltage
        b[source.n0] = 0
        x = torch.linalg.solve(A, b)
        for el in elements:
            el.I(x[el.n1], x[el.n0])
    if tracking:
        return tracking.I_values
    else:
        raise ValueError("No tracking element found.")


def run_simulation(
    epochs,
    source,
    elements,
    training_parameters,
    number_of_nodes,
    timesteps,
    sweep_time,
    target,
    device,
):
    target = target.to(device)
    optimizer = torch.optim.Adam(training_parameters)
    criterion = torch.nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        sim_output = sim(
            source, elements, number_of_nodes, timesteps, sweep_time, device
        )
        output = torch.stack(sim_output).squeeze()
        loss = criterion(output, target)
        if epoch % 20 == 0 or epoch == (epochs - 1):
            param_str = ", ".join(
                f"{group.get('name', 'unnamed')} = {group['params'][0].item():.4f}"
                for group in training_parameters
            )
            print(f"Epoch {epoch:3d} | {param_str}  | Loss = {loss.item():.3e}")
        loss.backward(retain_graph=True)
        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="simple.txt")
    parser.add_argument("epochs", nargs="?", type=int, default=1000)
    args = parser.parse_args()
    source, elements, parameters, number_of_nodes, timesteps, sweep_time = parse_source(
        "schematics/" + args.source, device
    )
    target = parse_target("targets/c_setup.txt", timesteps)
    run_simulation(
        args.epochs,
        source,
        elements,
        parameters,
        number_of_nodes,
        timesteps,
        sweep_time,
        target,
        device,
    )


if __name__ == "__main__":
    main()
