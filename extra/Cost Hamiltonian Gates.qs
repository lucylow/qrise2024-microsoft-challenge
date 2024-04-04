namespace Quantum {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Convert;

    function SetBitValue(reg: Int, bit: Int, value: Bool): Int {
        if(value) {
            return reg ||| (1 <<< bit);
        } else {
            return reg &&& ~~~(1 <<< bit);
        }
    }
    
    operation Circuit() : Unit {
        using(qubits = Qubit[4]) {
            Rz(50.725, qubits[0]);
            Rz(181.16, qubits[1]);
            Rz(-26.57, qubits[2]);
            Rz(-205.31, qubits[3]);
            zz(12.077, qubits[0], qubits[1]);
            zz(26.57, qubits[0], qubits[2]);
            zz(12.077, qubits[0], qubits[3]);
            zz(132.85, qubits[1], qubits[2]);
            ResetAll(qubits);
        }
    }
}
