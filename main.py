import argparse
from parse import parse_source, parse_target
import torch
from utils import plot_target_vs_output


def sim(source, elements, number_of_nodes, timesteps, sweep_time, device, hyst):
    """
    The sim function runs the linear sweep of a source and finds the IV curve. Currently all the implemented elements are linear which means we can solve using linalg.solve. However, once we add BJTs which aren't linear,
    we wil need to adjust this to use some type of other solver. Maybe Newton-Raphson? It works well for now though.
    """
    v_i = source.start
    v_f = source.end
    half = timesteps // 2 + 1
    if hyst:
        up = torch.linspace(v_i, v_f, half, device=device)
        down = torch.linspace(v_f, v_i, half, device=device)[1:]
        voltages = torch.cat([up, down])
    else:
        voltages = torch.linspace(v_i, v_f, int(timesteps), device=device)
    for i, voltage in enumerate(voltages):
        A = torch.zeros((number_of_nodes, number_of_nodes), device=device)
        b = torch.zeros((number_of_nodes, 1), device=device)
        tracking = None
        for element in elements:
            if i == 0:
                element.reset()
            A, b = element.G(A, b)
            if element.track:
                if tracking is None:
                    tracking = element
                else:
                    print(
                        f"Warning: Multiple tracking elements found. Using {tracking.name} for tracking."
                    )
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
    hyst,
):
    """Contains the main logic for running the simulation and optimizing the correct parameters"""
    target = target.to(device)
    optimizer = torch.optim.Adam(training_parameters)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        sim_output = sim(
            source, elements, number_of_nodes, timesteps, sweep_time, device, hyst
        )
        output = torch.stack(sim_output).squeeze()
        loss = criterion(output, target)
        if epoch % 2 == 0 or epoch == (epochs - 1):
            param_str = ", ".join(
                f"{group.get('name', 'unnamed')} = {group['params'][0].item():.4f}"
                for group in training_parameters
            )
            plot_target_vs_output(target, output, epoch, hyst, source)
            print(f"Epoch {epoch:3d} | {param_str}  | Loss = {loss.item():.3e}")
        loss.backward(retain_graph=True)

        optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="simple.txt")
    parser.add_argument("epochs", nargs="?", type=int, default=880)
    args = parser.parse_args()
    source, elements, parameters, number_of_nodes, timesteps, sweep_time, hyst = (
        parse_source("schematics/" + args.source, device)
    )
    target = parse_target("targets/c_tricky_hyst.txt", timesteps)
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
        hyst,
    )


if __name__ == "__main__":
    main()
