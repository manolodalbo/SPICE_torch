import argparse
from elements.resistor import Resistor
from elements.cap import Cap
from parse import parse_source, parse_target
import torch


def sim(source, elements, number_of_nodes, timesteps, sweep_time):
    A = torch.zeros((number_of_nodes, number_of_nodes))
    b = torch.zeros((number_of_nodes, 1))
    tracking = None
    for element in elements:
        if isinstance(element, Resistor):
            element.I_values = []
            one, two = element.G()

            A[one[0], one[0]] += one[1]
            A[one[0], two[0]] -= two[1]
            A[two[0], two[0]] += two[1]
            A[two[0], one[0]] -= one[1]
            if element.track:
                if tracking is None:
                    tracking = element
                else:
                    print(
                        f"Warning: Multiple tracking elements found. Using {tracking.name} for tracking."
                    )
        elif isinstance(element, Cap):
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
    v_i = source.start
    v_f = source.end
    voltages = torch.linspace(v_i, v_f, int(timesteps))
    for voltage in voltages:
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
    learning_rate,
    source,
    elements,
    training_parameters,
    number_of_nodes,
    timesteps,
    sweep_time,
    target,
):
    optimizer = torch.optim.Adam(training_parameters, lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        sim_output = sim(source, elements, number_of_nodes, timesteps, sweep_time)
        output = torch.stack(sim_output).squeeze()
        loss = criterion(output, target)
        if epoch % 20 == 0 or epoch == (epochs - 1):
            print(
                f"Epoch {epoch:3d} | R = {training_parameters} Ω | Loss = {loss.item():.6f}"
            )
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="simple.txt")
    parser.add_argument("epochs", nargs="?", type=int, default=1000)
    parser.add_argument("lr", nargs="?", type=float, default=0.1)
    args = parser.parse_args()
    source, elements, parameters, number_of_nodes, timesteps, sweep_time = parse_source(
        "schematics/" + args.source
    )
    target = parse_target("targets/r20.txt", timesteps)
    run_simulation(
        args.epochs,
        args.lr,
        source,
        elements,
        parameters,
        number_of_nodes,
        timesteps,
        sweep_time,
        target,
    )


if __name__ == "__main__":
    main()
