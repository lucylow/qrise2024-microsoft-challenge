/// # Description
/// This is a program for simulating the QAOA circuit. 
namespace qaoa{

    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Measurement;

    // Function for getting flat index
    operation flat_index(n: Int, i: Int, j: Int): Int{
        return n*i + j
    }

    operation cost_unitary(qubits: Qubit[], gamma: Double, quadratics: Double[], linears: Double[]): Unit{
        
        let n_qubits = Length(linears);
        mutable quad_sum : Double = 0.0;

        // RZ Gates
        for qubit in 0..n_qubits-1{
            set quad_sum = 0.0;
            for quad_qubit in 0..n_qubits-1{
                set quad_sum += quadratics[flat_index(n_qubits,qubit,quad_qubit)];
            }
            Rz(0.5 * (linears[qubit] + quad_sum) * gamma, qubits[qubit])
        }
        // RZZ Gates
        for i in 0..n_qubits-1{
            for j in i+1..n_qubits-1{                
                Rzz(0.25 * quadratics[flat_index(n_qubits,i,j)] * gamma, qubits[i], qubits[j])
            }
        }
    }

    operation mixer_unitary(qubits: Qubit[], beta: Double) : Unit{
        for qubit in qubits{
            Rx(2.0 * beta,qubit);
        }
    }

    operation circuit(NQubits: Int, Layers: Int, gammas: Double[], betas: Double[], quadratics: Double[], linears: Double[]) : Int {
        // Message("Q# Ciruit called");
        use q = Qubit[NQubits]; 
        mutable integer_result = 0;
        ApplyToEachA(H,q);
        // Message($"quad {quadratics}");
        // Message($"lin {linears}");

        for layer in 0..Layers-1{
            // Message($"Layer {i}");
            mixer_unitary(q, betas[layer]);
            cost_unitary(q, gammas[layer], quadratics, linears)
            
        }
        // for i in 0..NQubits-1{
        //     let r = M(q[i]);
        //     // Message($"{r}");
            
        //     if r == One{
        //         set integer_result += 2^i;  
        //     }
        // }

        // ResetAll(q);    
            
        // return integer_result;
        return MeasureInteger(q);
    }
}
