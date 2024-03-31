# Quantum Approximate Optimization Algorithm using Gate-based QC

# This script acts as a host for Q# to implement Quantum Approximate Optimization Algorithm (QAOA) on a gate-based Quantum Computing model.

# Importing required libraries

# General imports
import time
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt

# Libraries for Model Formulation
from docplex.mp.model import Model
from scipy.optimize import minimize
from collections import Counter

import qsharp
qsharp.init(project_root = 'C:/Users/londh/qc/QRISE_QRE/qaoa')

# Qiskit Imports
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.visualization import plot_histogram
from qiskit_optimization.translators import from_docplex_mp

# Library for circuit simulation
import pennylane as qml

# Function to sort the count dictionary.
def find_most_common_solutions(input_dict, n):
    sorted_keys = sorted(input_dict, key=input_dict.get, reverse=True)
    return sorted_keys[:n]


# Building the model and its Cost Function
 
# We are using ***docplex*** to build the model and calculate $Q$ and $c$.

def build_qubo(arr: list):

    n = len(arr)
    c = sum(arr)
    # Building the model and its QUBO formulation.
    model = Model()
    x = model.binary_var_list(n)

    Q =  (c - 2*sum(arr[i]*x[i] for i in range(n)))**2
    model.minimize(Q)   

    problem = from_docplex_mp(model)

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(problem)
    # print(qubo)

    quadratics = qubo.objective.quadratic.coefficients
    linears = qubo.objective.linear.coefficients

    return quadratics, linears, qubo

# ## Creating the QAOA circuit and layers.
# 
# I'm using ***Pennylane*** to handle the circuit simulations. I have created functions to generate the QAOA circuit given $Q$ and $c$.


# %%
def qaoa(arr,layers:int):

    quadratics, linears, qubo = build_qubo(arr)
    num_qubits = len(arr)

    # Initial guess
    init_gamma = np.array([pi/2]*layers)
    init_beta = np.array([pi/4]*layers)
    initial_guess = np.concatenate((init_gamma, init_beta))
    
    def expectation_value(theta):

        middle = int(len(theta)/2)
        gammas = theta[:middle]
        betas = theta[middle:]

        counts = qsharp.run("qaoa.circuit()",shots=1)
        best_sol = max(counts, key=counts.get)
        return qubo.objective.evaluate(np.array(list(best_sol), dtype='int'))


    # Minimization of the objective function.
    start_time = time.time()
    res = minimize(expectation_value, initial_guess, method='COBYLA')
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'Elapsed time for QAOA: {elapsed_time} seconds')

    middle = int(len(res.x)/2)
    prime_gammas = res.x[:middle]
    prime_betas = res.x[middle:]

    counts = qaoa_circuit_generator(num_qubits, layers,prime_gammas, prime_betas, quadratics, linears)
    
    return counts

test_arr = [12,4,6]

n_qubits = len(test_arr)
layers = 3

gamma = [0.2]
beta = [0.2]

input_str = str(n_qubits) + "," + str(layers) + "," + "[0.4]" + "," + "[0.2]" + "," + "[[0.3,0.23],[0.9,0.85]]" + "," + "[0.5]"

counts = qsharp.run("qaoa.circuit(" + input_str + ")",shots=1)

print(counts)
# print(counts[0][0],type(counts[0][0]))