# Quantum Approximate Optimization Algorithm (QAOA)

# This script is a PennyLane implementation of the QAOA algorithm.
# Importing required libraries

# General imports
import time
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt

# Libraries for Model Formulation
from docplex.mp.model import Model
from scipy.optimize import minimize

# Qiskit Imports
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

# Library for circuit simulation
import pennylane as qml

# Defining helper functions.
def find_most_common_solutions(input_dict, n):
    """
    Sorts the keys of the input dictionary in descending order based on their values and returns the first n keys.

    Parameters:
        input_dict (dict): A dictionary containing the keys and their corresponding values.
        n (int): The number of most common solutions to return.

    Returns:
        list: A list of the n most common keys sorted in descending order based on their values.
    """
    sorted_keys = sorted(input_dict, key=input_dict.get, reverse=True)
    return sorted_keys[:n]


# Building the model and its Cost Function
def build_qubo(arr: list):
    """
    Function to build a QUBO (Quadratic Unconstrained Binary Optimization) model from a given array
    for the Number Partitioning Problem (NPP).

    :param arr: a list of integers representing the array from which the QUBO model is built
    :return: a tuple containing the quadratic coefficients, linear coefficients, and the QUBO model
    """

    # Length of the array - Length of the binary vector x
    n = len(arr)
    # Sum of the array - c
    c = sum(arr)

    # Building the model and its QUBO formulation.
    model = Model()
    x = model.binary_var_list(n)

    # Cost Function for Number Partirioning Problem (NPP)
    Q =  (c - 2*sum(arr[i]*x[i] for i in range(n)))**2
    model.minimize(Q)   
    problem = from_docplex_mp(model)

    # QUBO formulation
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(problem)

    # Quadratic and Linear Coefficients
    quadratics = qubo.objective.quadratic.coefficients
    linears = qubo.objective.linear.coefficients

    return quadratics, linears, qubo

# Creating the QAOA circuit and layers.
# Defining the Cost and the Layers of QAOA.
# Cost Layer.
def U_C(gamma,quadratics,linears,num_qubits):

    for wire in range(num_qubits):
        qml.RZ(1/2*(linears[(0, wire)]+sum(quadratics[(wire, j_wire)] for j_wire in range(num_qubits)))*gamma,wires=wire)

    for (wire1, wire2) in quadratics.keys():
        if wire1!=wire2:
            qml.CNOT(wires=[wire1, wire2])
            qml.RZ(1/4*quadratics[(wire1, wire2)]*gamma,wires=wire2)
            qml.CNOT(wires=[wire1, wire2])

# Mixer Layer.
def U_M(beta,num_qubits):
    for wire in range(num_qubits):
        qml.RX(2*beta,wires=wire)

# Function to generate the QAOA circuit given parameters and coefficients.
def qaoa_circuit_generator(num_qubits,layers,gammas,betas,quadratics,linears):
    dev = qml.device("lightning.qubit", wires=num_qubits, shots=1024)
    @qml.qnode(dev)
    def circuit(gammas,betas,quadratics,linears):

        for qubit in range(num_qubits):
            qml.Hadamard(wires=qubit)
        qml.Barrier()

        for layer in range(layers):
            U_C(gammas[layer],quadratics,linears,num_qubits)
            qml.Barrier()
            U_M(betas[layer],num_qubits)
            qml.Barrier()
        
        return qml.counts(wires=range(num_qubits))

    return circuit(gammas,betas,quadratics,linears)

# Global Variables for tracking.
func_call = 0
theta = []
cost = []

# Callback fucntion for the optimizer
def callback_func(x):
    theta.append(x)

def qaoa(arr,layers:int):
    """
    Applies the Quantum Approximate Optimization Algorithm (QAOA) to solve the Quadratic Unconstrained Binary Optimization (QUBO) problem.

    Parameters:
    - arr (list): The input array.
    - layers (int): The number of layers in the QAOA circuit.

    Returns:
    - counts (dict): A dictionary representing the counts of the solution.

    Note:
    - This function uses the `build_qubo` function to generate the quadratics, linears, and QUBO objects.
    - The `expectation_value` function is defined to calculate the expectation value of the QAOA circuit.
    - The `minimize` function from the `scipy.optimize` module is used to minimize the objective function.
    - The QAOA circuit is generated using the `qaoa_circuit_generator` function to generate the QAOA circuit.

    Example:
    ```
    arr = [1, 2, 3]
    layers = 3
    counts = qaoa(arr, layers)
    ```
    """

    quadratics, linears, qubo = build_qubo(arr)
    num_qubits = len(arr)

    # Initial guess
    init_gamma = np.array([pi/1.5]*layers)
    init_beta = np.array([pi/4]*layers)

    initial_guess = np.concatenate((init_gamma, init_beta))
    
    def expectation_value(theta):
        # print('expectation_value called')
        global func_call 
        func_call = func_call + 1

        middle = int(len(theta)/2)
        gammas = theta[:middle]
        betas = theta[middle:]

        # Calling the QAOA circuit.
        counts = qaoa_circuit_generator(num_qubits, layers, gammas, betas, quadratics, linears)
        best_sol = max(counts, key=counts.get)

        exp = qubo.objective.evaluate(np.array(list(best_sol), dtype='int'))
        cost.append(exp)

        print(f'Function call: {func_call} - Cost: {exp}')

        return exp


    # Minimization of the objective function.
    start_time = time.time()
    res = minimize(expectation_value, initial_guess, method='COBYLA',callback=callback_func)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'\nElapsed time for QAOA: {elapsed_time} seconds')

    middle = int(len(res.x)/2)
    prime_gammas = res.x[:middle]
    prime_betas = res.x[middle:]

    counts = qaoa_circuit_generator(num_qubits, layers,prime_gammas, prime_betas, quadratics, linears)
    
    return counts

# Defining a test array
test_array = [5,4,6,1,12,7]
layers = 4

# Running QAOA on for Number Partitioning. 
counts = qaoa(test_array,layers)

# Plotting the output state.
plt.figure(figsize=(15, 5))
plt.bar(range(len(counts)), list(counts.values()), align='center', color='red')
plt.xticks(range(len(counts)), list(counts.keys()), rotation=90)
plt.title("QAOA Output State")
plt.xlabel("Bitstrings")
plt.ylabel("Counts")
plt.grid()
plt.show()

# Plotting Cost vs. iterations.
plt.figure(figsize=(15, 5))
plt.plot(range(len(cost)),cost,color='g',ls='--',marker='o',lw=2)
plt.xticks(range(1,len(cost)+1,5))
plt.title('Cost vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.grid()
plt.show()

best_sol = find_most_common_solutions(counts,3)
print(f'\nQAOA Top 3 solutions for {test_array} and {layers} layers: \n{best_sol}')

c = sum(test_array)

S = []
S_A = []
for ind,bit in enumerate(best_sol[0]):
    if bit == '1':
        S.append(ind)
    else:
        S_A.append(ind)

sum_S = sum(np.array(test_array)[S])
sum_S_A = sum(np.array(test_array)[S_A])    

print(f'\n\n Best partition:\nS {np.array(test_array)[S]} - Sum(S) = {sum_S}\nS/A {np.array(test_array)[S_A]} - Sum(S/A) = {sum_S_A}')
