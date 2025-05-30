from elements.resistor import Resistor
from elements.cap import Cap
from elements.v_source import VSource

import torch


def parse_source(source):
    elements = []
    nodes_seen = set()
    maximum_node = 0
    parameters = []
    with open(source, "r") as f:
        for index, line in enumerate(f):
            line = line.strip()
            line = line.split()
            if index == 0:
                timesteps = int(line[0])
                sweep_time = float(line[1])
                timestep = sweep_time / timesteps
            else:
                el = "".join([char for char in line[0] if not char.isdigit()])
                if el == "R":
                    name = line[0]
                    n0 = int(line[2])
                    n1 = int(line[3])
                    nodes_seen.add(n0)
                    nodes_seen.add(n1)
                    maximum_node = max(maximum_node, n0, n1)
                    resistance = float(line[1])
                    if len(line) > 4:
                        track = True
                    else:
                        track = False
                    resistor = Resistor(name, resistance, n0, n1, track)
                    elements.append(resistor)
                    print(
                        f"Created Resistor: {resistor.name} with R={resistor.R} between nodes {resistor.n0} and {resistor.n1}"
                    )
                elif el == "C":
                    name = line[0]
                    n0 = int(line[2])
                    n1 = int(line[3])
                    nodes_seen.add(n0)
                    nodes_seen.add(n1)
                    maximum_node = max(maximum_node, n0, n1)
                    capacitance = float(line[2])
                    timestep = float(line[4])
                    if len(line) > 5:
                        track = True
                    else:
                        track = False
                    cap = Cap(name, capacitance, n0, n1, timestep, track)
                    elements.append(cap)
                    print(
                        f"Created Capacitor: {cap.name} with C={cap.C} between nodes {cap.n0} and {cap.n1} with timestep {cap.timestep}"
                    )
                elif el == "V":
                    name = line[0]
                    n0 = int(line[3])
                    n1 = int(line[4])
                    nodes_seen.add(n0)
                    nodes_seen.add(n1)
                    maximum_node = max(maximum_node, n0, n1)
                    start = float(line[1])
                    end = float(line[2])
                    timesteps = int(line[5])
                    vsource = VSource(name, start, end, n0, n1)
                    source = vsource
                    print(
                        f"Created Voltage Source: {vsource.name} with start={vsource.start}, end={vsource.end} between nodes {vsource.n0} and {vsource.n1}"
                    )
                else:
                    print(f"Unknown element type: {el}")
    number_of_nodes = len(nodes_seen)
    if maximum_node >= number_of_nodes:
        print(
            f"Warning: Maximum node number {maximum_node} exceeds the number of unique nodes {number_of_nodes}."
        )
        exit(1)
    return source, elements, number_of_nodes, timesteps, sweep_time


def parse_target(target):
    with open(target, "r") as f:
        lines = f.readlines()
    target_data = []
    for line in lines:
        line = line.strip()
        if line:
            values = list(map(float, line.split()))
            target_data.append(values)
    target_tensor = torch.tensor(target_data, dtype=torch.float32)
    return target_tensor
