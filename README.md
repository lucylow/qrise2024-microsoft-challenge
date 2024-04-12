# Microsoft Quantum Research Challenge at QRISE 2024: Quantum Resource Estimation of QAOA

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/qrise2024-microsoft-challenge.svg)](https://github.com/lucylow/qrise2024-microsoft-challenge/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/qrise2024-microsoft-challenge.svg)](https://github.com/lucylow/qrise2024-microsoft-challenge/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

![](https://github.com/lucylow/qrise2024-microsoft-challenge/blob/main/extra/Logo.png?raw=true)


## Prerequisites
Ensure you have the following prerequisites installed:
- Visual Studio Code (latest version) or open VS Code on the Web.
- Azure Quantum Development Kit extension (latest version).

## Introduction
This repository contains code examples and instructions for using the Azure Quantum Resource Estimator to analyze resource requirements for QUBO (Quadratic Unconstrained Binary Optimization) numerical partitioning problems.

### Load a QUBO Sample Program
Follow these steps to load a QUBO sample program:
1. In VS Code, select File > New File and save the file as `RandomNum.qs`.
2. Open `RandomNum.qs` and type `sample`, then select `Random Bit sample` and save the file.

### Run the Resource Estimator
Execute the following steps to run the Resource Estimator:
1. Open the Resource Estimator window by selecting View -> Command Palette or pressing Ctrl+Shift+P. Then type “resource” to bring up the option "Q#: Calculate Resource Estimates" and select it.
2. Choose one or more Qubit parameter + Error Correction code types to estimate the resources. For this example, select `qubit_gate_us_e3` and click OK.
3. Specify the Error budget or accept the default value (0.001).
4. Press Enter to accept the default result name based on the filename, which in this case is `RandomNum`.

### Parameter Customization
1. Select Target Parameters:
   - Choose the desired qubit parameter and error correction code combination. For this example, select "qubit_gate_us_e3" and "surface_code."
   - Click OK to proceed.
2. Specify Error Budget:
   - Optionally, specify the error budget. You can either accept the default value (0.001) or enter a new one.
   - Press Enter to continue.
3. Change Target Parameters:
   - To estimate costs for different configurations, repeat the process by selecting alternative qubit types, error correction codes, and error budgets.
4. Run Multiple Configurations of Parameters:
   - Open the Resource Estimator window again.
   - Select multiple configurations of target parameters and compare the resource estimation results.
   - The Space-time diagram and the table of results will display information for all selected configurations, allowing for comprehensive comparison.


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

### View the Results
Explore the results of the resource estimation:

- **Results Tab**: Provides a summary of the resource estimation. You can select the columns you want to display, including runtime, physical qubits, logical qubits, and more.
- **Space-time Diagram**: Illustrates the tradeoffs between the number of physical qubits and the runtime of the algorithm. Hover over each point to view detailed resource estimation.
- **Space Diagram Tab**: Displays the distribution of physical qubits used for the algorithm and T factories.
- **Resource Estimates Tab**: Presents the full list of output data, including details on QEC scheme, code distance, physical qubits, logical cycle time, and error rates.
   - Customize the displayed columns by clicking the icon next to the first row and selecting from various options such as run name, estimate type, qubit type, qec scheme, error budget, logical qubits, and more. Explore various configurations to understand resource requirements for different QUBO problems effectively.

### Next Steps
Continue exploring the capabilities of the Resource Estimator:
- Experiment with different SDKs and IDEs.
- Learn how to leverage the Resource Estimator for various QUBO problems and scenarios.

## Further Information
For detailed instructions and code snippets, refer to the [full documentation](https://github.com/lucylow/qrise2024-microsoft-challenge/blob/main/qaoa/qaoa_notebook.ipynb) and [the writeup](https://github.com/lucylow/qrise2024-microsoft-challenge/blob/main/writeup.pdf).

## Contributors

- Katie Harrison (@Katie1harrison)
- Muhammad Waqar Amin (@Eagle-Eyes7)
- Sarah Dweik (@Sarah-Dweik)
- Nikhil Londhe (@nikhil-co)
- Lucy Low (@lucylow)

## License
This project is licensed under the [MIT License](link_to_license).

