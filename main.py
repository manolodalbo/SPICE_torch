import argparse
from elements.resistor import Resistor
from elements.cap import Cap
from elements.v_source import VSource
from parse import parse_source
import torch
from concurrent.futures import ThreadPoolExecutor

def sim(source, elements, number_of_nodes,timesteps,sweep_time):
    A = torch.zeros((number_of_nodes, number_of_nodes))
    b = torch.zeros((number_of_nodes, 1))
    tracking = None
    for element in elements:
        if isinstance(element, Resistor):
            one, two = element.G()
            A[one[0], one[0]] += one[1]
            A[two[0], two[0]] += two[1]
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
    A[source.n1, source.n1] = 1
    A[source.n1, source.n0] = -1
    v_i = source.start
    v_f = source.end
    voltages = torch.linspace(v_i, v_f, int(timesteps))
    for voltage in voltages:
        b[source.n1] = voltage
        b[source.n0] = 0
        x = torch.linalg.solve(A, b)
        with ThreadPoolExecutor() as executor:
            executor.map(lambda el: el.I(x[el.n1].item(), x[el.n0].item()), elements)
    if tracking:
        return tracking.I_values
    else:
        raise ValueError("No tracking element found.")
def run_simulation(epochs,learning_rate,training_parameters,source, elements, number_of_nodes, timesteps, sweep_time):
    optimizer = torch.optim.Adam(,lr=learning_rate)
    for step in range(epochs):


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", default="simple.txt")
    parser.add_argument("epochs",default=100)
    parser.add_argument("lr",default=0.001)
    args = parser.parse_args()
    source, elements, parameters, number_of_nodes, timesteps, sweep_time = parse_source(
        "schematics/" + args.source
    )
    run_simulation(args.epochs,args.lr,source, elements, number_of_nodes, timesteps, sweep_time)


if __name__ == "__main__":
    main()
