import pennylane as qml
from quorus.logging.custom_slog import print_cust
from functools import partial
from quorus.qnode_funcs.template_funcs.qcnn_template import qcnn_template

def create_qnode_qcnn(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", num_ancillas=0, layer_types_list=[], cheating=False, tunn_down=False):
    """
    Creates a QNode of the specified size.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN
      pennylane_interface: a string representing the interface that pennylane uses for this qnode.
      num_ancillas: an integer specifying the number of ancillas uised for this qnode.
      layer_types_list: a list of strings where the string at layer i specifies the ansatz type to use in this layer.
      cheating: a Boolean indicating whether or not cheating measurments should be applied.
      tunn_down: a Boolean indicating whether or not the circuit should have less layers for later layers in the circuit.

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    n_qubits = n_data
    dev = qml.device("default.qubit", wires=n_qubits)
    print_cust(f"create_qnode_qcnn, pennylane_interface: {pennylane_interface}, num_ancillas: {num_ancillas}, cheating: {cheating}, tunn_down: {tunn_down}")
    circuit = partial(qcnn_template, n_qubits=n_qubits, expansion_data=expansion_data, n_classes=n_classes, num_ancillas=num_ancillas, layer_types_list=layer_types_list, cheating=cheating, tunn_down=tunn_down)

    return qml.qnode(dev, interface=pennylane_interface)(circuit)