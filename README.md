# SPICEtorch

The purpose of SPICEtorch is to allow for the 'farming' of parameters to replicate a specified IV curve. This issue has arisen due to attempts to make SPICE simulations replicate the experimental IV curves of the transmemristor. Although initial guesses come close to experimental data, the ability for the farming of these parameters by a machine learning model helps to both get more accurate parameters and strengthen our confidence in the SPICE model.

## Methodology

The optimization of parameters is done via a gradient descent algorithm. The current implementation only works for configurations with resistors, capacitors, and a voltage source. Therefore, all valid circuits can be easily solved usign torch.linalg.solve as all elements are linear. The loss is calculated using MSE between the target currents at each timestep and Every 20 epochs, the loss and parameters optimized are printed out and the target vs generated output is plotted in the plots folder. The configuration of elements can be done via a txt file. Here is what the format should be for accurate parsing

txt file :
Keep in mind that node 0 is hardcoded to ground
1000 1 h <-the first line should first contain the number of timesteps to calculate for. The second number is the duration and finally, include an h for hysterysis if the algorithm should optimize both forward and backwards sweep
the subsequent lines can be in any order, and you must define different elements

For a voltage source the format is name(V followed by some numbers) starting_voltage upper_voltage negative_node positive_node

For a capacitor the format is name(C followed by some numbers) initial_capacitance_value node_1 node_2 (at the end add t to track the current through this element and/or o to optimize this capacitor)
For a resistor the format is name(R followed by some )
