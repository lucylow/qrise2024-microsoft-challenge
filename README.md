# Microsoft Quantum research challenge at QRISE 2024: Resource estimation of quantum algorithms 

QUBO Numerical Partitioning with Azure Quantum Resource Estimator

## Prerequisites
Ensure you have the following prerequisites installed:
- Visual Studio Code (latest version) or open VS Code on the Web.
- Azure Quantum Development Kit extension (latest version).

## Introduction
This repository contains code examples and instructions for using the Azure Quantum Resource Estimator to analyze resource requirements for QUBO (Quadratic Unconstrained Binary Optimization) numerical partitioning problems.

## Usage
1. Load a QUBO Sample Program:
   - Create a new file named `RandomNum.qs`.
   - Type `sample` in the file, then select `Random Bit sample` and save the file.

2. Quantum Intermediate Representation (QIR):
   - Azure Quantum Resource Estimator uses QIR, a standardized format for quantum programs.

3. Define a Function to Create a Resource Estimation Job from QIR:
   - Use the provided Python function to generate an Azure Quantum job from QIR.

4. Run a Sample Quantum Program:
   - Generate QIR bitcode using PyQIR generator.
   - Use the defined function to generate a resource estimation job.
   - Run the job and view the results.

5. Run the Resource Estimator:
   - Open the Resource Estimator window in VS Code.
   - Choose target parameters and error correction codes.
   - Specify error budget and run the estimation.

6. View the Results:
   - Explore the results of the resource estimation, including runtime, physical qubits, logical qubits, and more.
   - Analyze the Space-time diagram and Space diagram for resource distribution.

7. Next Steps:
   - Experiment with different SDKs and IDEs.
   - Leverage the Resource Estimator for various QUBO problems and scenarios.

## Further Information
For detailed instructions and code snippets, refer to the [full documentation](link_to_documentation).

## Contributors
- John Doe (@johndoe)
- Jane Smith (@janesmith)

## License
This project is licensed under the [MIT License](link_to_license).

