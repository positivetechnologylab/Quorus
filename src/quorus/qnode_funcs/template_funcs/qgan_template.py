from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.circuit_models.qgan_circuit import QGAN_circuit

"""## QGAN QNode Creation

### QNode QGAN, Global Function
"""

def qgan_template(input_noise, params, n_qubits):
    """
    Runs the quantum circuit for QGAN, given specific configurations.

    Parameters:
        input_noise: an array or tensor representing the inputs to be fed in to the model.
        params: a tuple of parameters to be called with this qnode.
        n_qubits: an integer representing the number of qubits for this QNode

    Returns:
        a tensor or list of tensors representing the output of running through this QNode.
    """
    # params should just be a list of tensors representing the parameters to be used in the QGAN.
    # this qnode should not be exposed externally; should only be used in the context of PatchQuantumGenerator.
    print_cust(f"qgan_template, input_noise: {input_noise}, n_qubits: {n_qubits}, params: {params}")
    # print_cust(f"qgan_template, qml.draw: {qml.draw(QGAN_circuit)(input_noise, n_qubits, params)}")
    return QGAN_circuit(input_noise, n_qubits, params)