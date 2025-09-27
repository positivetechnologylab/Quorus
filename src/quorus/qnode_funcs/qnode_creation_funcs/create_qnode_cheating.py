from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from functools import partial
from quorus.qnode_funcs.qnode_run_funcs.cheating_run import run_multiprob_qnode_cheating

def create_qnode_qcnn_multieval_cheating(n_data, conv_layers, expansion_data, n_classes=2, pennylane_interface="autograd", layer_types_list=[]):
    """
    Creates a multi-evaluation QNode of the specified size. Cheating measurement is applied (i.e., measuring without collapsing the quantum state)

    Parameters:
      n_data: an integer representing the number of qubits for this QNode
      conv_layers: an integer representing the number of convolutional layers in this QNode
      expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
      n_classes: an integer representing the number of output classes for this QCNN
      pennylane_interface: a string representing the interface that pennylane uses for this qnode.
      layer_types_list: a list of strings where the string at layer i specifies the ansatz type to use in this layer.

    Returns:
      qnode: a QNode object satisfying the above constraints.
    """
    qcnn_qnode = create_qnode_qcnn(n_data, conv_layers, expansion_data, n_classes=n_classes, pennylane_interface=pennylane_interface, layer_types_list=layer_types_list, cheating=True)
    # return a partial to run_multiprob_qnode
    multiprob_fn = partial(run_multiprob_qnode_cheating, qnode=qcnn_qnode)
    return multiprob_fn