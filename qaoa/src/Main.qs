/// # Description
/// This is a program for simulating the QAOA circuit. 
namespace qaoa {

    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Measurement;

    operation cost_unitary(qubits: Qubit[], gamma: Double, quadratics: Double[], linears: Double[]): Unit{
        for qubit in qubits{
            // Rz(0.1,qubit)
        }
        // Rzz
    }

    operation mixer_unitary(qubits: Qubit[], beta: Double) : Unit{
        for qubit in qubits{
            Rx(beta,qubit);
        }
    }

    operation circuit(NQubits: Int, Layers: Int, gammas: Double[], betas: Double[], quadratics: Double[][], linears: Double[]) : Result[] {
        use q = Qubit[NQubits]; 

        // Message($"quad {quadratics}");

        // Message($"{IsSquareArray(quadratics)}");
        
        for i in 1..Layers {
            Message($"Layer {i}");
            mixer_unitary(q, 0.5);
            // cost_unitary(q,gammas[i],quadratics,linears)
        }
        return MResetEachZ(q);
    }
}
