from elements.resistor import Resistor
from elements.cap import Cap
from elements.v_source import VSource
import numpy as np
import torch


def track_and_train(line, start_index):
    track = False
    train = False
    if len(line) > start_index:
        if line[start_index] == "t":
            track = True
        elif line[start_index] == "o":
            train = True
    if len(line) > start_index + 1:
        if line[start_index + 1] == "t":
            track = True
        elif line[start_index + 1] == "o":
            train = True
    return track, train


def parse_source(source, device):
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
                    track, train = track_and_train(line, 4)
                    name = line[0]
                    n0 = int(line[2])
                    n1 = int(line[3])
                    nodes_seen.add(n0)
                    nodes_seen.add(n1)
                    maximum_node = max(maximum_node, n0, n1)
                    resistance = float(line[1])
                    resistor = Resistor(
                        name, resistance, n0, n1, timesteps, device, track, train
                    )
                    resistor.lr = resistor.R / 10
                    elements.append(resistor)
                    if resistor.opt:
                        parameters.append(
                            {
                                "params": [resistor.R],
                                "lr": resistor.get_lr(),
                                "name": resistor.name,
                            }
                        )
                elif el == "C":
                    track, train = track_and_train(line, 4)
                    name = line[0]
                    n0 = int(line[2])
                    n1 = int(line[3])
                    nodes_seen.add(n0)
                    nodes_seen.add(n1)
                    maximum_node = max(maximum_node, n0, n1)
                    capacitance = float(line[1])
                    cap = Cap(
                        name,
                        capacitance,
                        n0,
                        n1,
                        timestep,
                        timesteps,
                        device,
                        track,
                        train,
                    )
                    cap.lr = cap.C / 10
                    elements.append(cap)
                    if cap.opt:
                        parameters.append(
                            {
                                "params": [cap.C],
                                "lr": cap.get_lr(),
                                "name": cap.name,
                            }
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
                    vsource = VSource(name, start, end, n0, n1)
                    source = vsource
                else:
                    print(f"Unknown element type: {el}")
    number_of_nodes = len(nodes_seen)
    if maximum_node >= number_of_nodes:
        print(
            f"Warning: Maximum node number {maximum_node} exceeds the number of unique nodes {number_of_nodes}."
        )
        exit(1)
    return source, elements, parameters, number_of_nodes, timesteps, sweep_time


def parse_ltspice_txt(filename):
    """
    Parses an LTspice-exported .txt file with two columns.

    Args:
        filename (str): Path to the .txt file

    Returns:
        time (np.ndarray): Time values
        values (np.ndarray): Corresponding column (e.g., I(R1))
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Skip header
    data = [line.strip().split() for line in lines[1:] if line.strip()]

    # Convert to float arrays
    time = np.array([float(row[0]) for row in data])
    values = np.array([float(row[1]) for row in data])

    return time, values


def resample_signal(time, values, timesteps):
    """
    Linearly interpolates LTspice signal data to match a given number of time steps.

    Args:
        time (np.ndarray): Original time values
        values (np.ndarray): Original signal values (e.g., I(R1))
        timesteps (int): Desired number of uniformly spaced time steps

    Returns:
        time_uniform (np.ndarray): Uniformly spaced time points
        values_uniform (np.ndarray): Interpolated values at those time points
    """
    time_uniform = np.linspace(time[0], time[-1], timesteps)
    values_uniform = np.interp(time_uniform, time, values)
    return time_uniform, values_uniform


def parse_target(target, timesteps):
    time, values = parse_ltspice_txt(target)
    t, values_uniform = resample_signal(time, values, timesteps)
    return torch.from_numpy(values_uniform).float()
