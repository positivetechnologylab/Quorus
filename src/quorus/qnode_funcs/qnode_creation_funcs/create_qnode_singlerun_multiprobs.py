from quorus.logging.custom_slog import print_cust
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from functools import partial
from quorus.qnode_funcs.qnode_run_funcs.singleeval_run import run_singleeval_qnode

def create_qnode_qcnn_singleeval(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", block_layers=5, layer_types_list=[]):
    """
    Creates a multi-evaluation QNode of the specified size. Runs the QNode once, but with ancilla qubits.

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN
      pennylane_interface: a string representing the interface that pennylane uses for this qnode.
      block_layers: an integer specifying the number of number of block layers used in the circuit. NOTE: this feature
      is not complete; currently, it is hardcoded to always be 5.
      layer_types_list: a list of strings where the string at layer i specifies the ansatz type to use in this layer.

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    # TODO, public code: yes, it is currently a hack that block_layers is 5 here; it should be injected.
    # TOMODIFY, DepthFL, HACK: inject the number of additional ancillas that I want my qnode to have.
    # ^ currently, is hardcoded as a default arg to see if it works.
    print_cust(f"create_qnode_qcnn_singleeval, n_data: {n_data}, block_layers: {block_layers}")
    # TOMODIFY, DepthFL: for now, assuming (or maybe not? it's just named this way) that number of
    # block layers = number of ancillas in the circuit.
    qcnn_qnode = create_qnode_qcnn(n_data + block_layers, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, num_ancillas=block_layers, layer_types_list=layer_types_list)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_singleeval_qnode, qnode=qcnn_qnode)
    return multiprob_fn