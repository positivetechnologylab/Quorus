"""## QCNN QNode Creation"""

from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.circuit_models.qcnn_circuit import QCNN_circuit_dynamic

def qcnn_template(input_angles, params, n_qubits, expansion_data, n_classes, num_ancillas=0, layer_types_list=[], cheating=False, tunn_down=False):
  """
  Runs the quantum circuit for QCNN, given specific configurations.

  Parameters:
    input_angles: the data fed into the quantum model of interest
    params: a tuple of parameters to be called with this qnode.
    n_qubits: an integer representing the number of qubits for this QNode
    expansion_data: a list of (encoded_angles, reversed_indices) that is used for identity initialization
    n_classes: an integer representing the number of output classes for this QCNN
    num_ancillas: an integer specifying the number of ancillas uised for this qnode.
    layer_types_list: a list of strings where the string at layer i specifies the ansatz type to use in this layer.
    cheating: a Boolean indicating whether or not cheating measurments should be applied.
    tunn_down: a Boolean indicating whether or not the circuit should have less layers for later layers in the circuit.

  Returns:
    a tensor or list of tensors representing the output of running through this QNode.
  """
  # TOMODIFY, depthFL: add num_ancillas to what this function calls.
  print_cust(f"qcnn_template, input_angles: {input_angles}, params: {params}, n_qubits: {n_qubits}, expansion_data: {expansion_data}, n_classes: {n_classes}, num_ancillas: {num_ancillas}, layer_types_list: {layer_types_list}, cheating: {cheating}, tunn_down: {tunn_down}")
  # print_cust(f"qcnn_template, qml.draw: {qml.draw(QCNN_circuit_dynamic)(input_angles, params[0], params[1], params[2], params[3], n_qubits, expansion_data=expansion_data, block_params_list=params[5], n_classes=n_classes, num_ancillas=num_ancillas, layer_types_list=layer_types_list, cheating=cheating, tunn_down=tunn_down)}")
  return QCNN_circuit_dynamic(input_angles, params[0], params[1], params[2], params[3],
                              n_qubits, expansion_data=expansion_data, block_params_list=params[5], n_classes=n_classes, num_ancillas=num_ancillas, layer_types_list=layer_types_list, cheating=cheating, tunn_down=tunn_down)