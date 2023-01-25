# Zero Noise Extrapolation (ZNE)
The idea of zero noise extrapolation relies on the assumption that it is possible to increase the strength of noise in a quantum circuit, e.g., by introducing additional gates. 

 Since the main errors originate from imperfect CNOT gates, the simplest method is to replace each CNOT gate by 3 CNOT gates.

We will demonstrate the zero noise extrapolation for the HHL-algorithm with 4 qubits, see https://qiskit.org/textbook/ch-applications/hhl_tutorial.html. This algorithms solves a two-dimensional, linear system of equations. The function 𝐹 that we are interested in is the norm of the corresponding solution.

The module ```algorithm``` contains the function algo generating the quantum circuit of the HHL algorithm and the function eval_counts corresponding to the above function 𝐹. The module ```zne``` contains the class ZNE needed to perform the zero noise extrapolation.

[Here](https://gitlab.cc-asp.fraunhofer.de/koenig1/ZNE/-/blob/main/example.ipynb) or under ```example_zne.ipynb``` is an example notebook with further explanations.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SEQUOIA-Demonstrators/zne_sequoia/HEAD)
