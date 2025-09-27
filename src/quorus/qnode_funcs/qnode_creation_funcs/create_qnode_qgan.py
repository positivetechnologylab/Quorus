import pennylane as qml
from functools import partial
from quorus.qnode_funcs.template_funcs.qgan_template import qgan_template

def create_qnode_qgan(n_data):
    """
    Creates a QNode of the specified size, for QGAN.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    n_qubits = n_data
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = partial(qgan_template, n_qubits=n_qubits)
    # @qml.qnode(dev, interface="torch")
    # def qnode_qgan(input_noise, n_qubits, params):
    #     # params should just be a list of tensors representing the parameters to be used in the QGAN.
    #     # this qnode should not be exposed externally; should only be used in the context of PatchQuantumGenerator.
    #     print_cust(f"qnode_qgan, qml.draw: {qml.draw(QGAN_circuit)(input_noise, n_qubits, params)}")
    #     return QGAN_circuit(input_noise, n_qubits, params)
    # return qnode_qgan
    return qml.qnode(dev, interface="torch")(circuit)