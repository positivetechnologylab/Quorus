"""## QCNN Circuit"""

from functools import partial

from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.data_encoding.data_encoding_funcs import pennylane_angleencode
from quorus.quantum_circuit_funcs.circuit_models_components.variational_layers_func import block_variational_circuit
from quorus.quantum_circuit_funcs.utils.utils import unpermute_inputs
from quorus.quantum_circuit_funcs.circuit_models_components.qcnn_funcs import conv_layer, pool_layer_with_measurement
from pennylane import numpy as np
import pennylane as qml

def QCNN_circuit_dynamic(inputs, conv_params_tuple, pool_params_tuple, final_pool_param, final_params, n_qubits, n_classes=2, expansion_data=[], block_params_list=[], num_ancillas=0, layer_types_list=[], cheating=False, tunn_down=False):
    """
    Dynamic QCNN circuit with optional variational blocks.
    Parameters:
      inputs: Numpy array representing the input data to the circuit.
      conv_params_tuple: Numpy array representing the convolutional parameters for the circuit.
      pool_params_tuple: Numpy array representing the pooling parameters for the circuit.
      final_pool_param: Numpy array representing the final pool parameter for the circuit.
      final_params: Numpy array representing the final parameters used in the circuit.
      n_qubits: Integer representing the number of qubits in the circuit.
      n_classes: Integer represnting the number of classes to classify for. (currently, in the code, it is assumed that n_output_qubits = int(np.ceil(np.log2(n_classes)))).
      expansion_data: A list of (encoded_angles, reversed_indices) for use for maintaining an identity initialization for this QCNN when the circuit expands, if supplied.
      block_params_list: list of block parameter arrays. Each array should have shape
        (num_layers_bp, num_qubits_bp, 3). If nonempty, then as the circuit is built,
        whenever the current number of wires equals num_qubits_bp, the block is applied.
      num_ancillas: an integer specifying the number of ancillas uised for this qnode.
      layer_types_list: a list of strings where the string at layer i specifies the ansatz type to use in this layer.
      cheating: a Boolean indicating whether or not cheating measurments should be applied.
      tunn_down: a Boolean indicating whether or not the circuit should have less layers for later layers in the circuit.

      Returns:
        The probability of measuring the amplitudes for the remaining qubits.
    """
    # Data embedding
    # print_cust(f"QCNN_circuit_dynamic, inputs: {inputs}")
    # assume that the input data is permuted correctly to fit linearly on the qubits.
    print_cust(f"QCNN_circuit_dynamic, n_qubits: {n_qubits}, num_ancillas: {num_ancillas}")

    print_cust(f"QCNN_circuit_dynamic, cheating: {cheating}")

    print_cust(f"QCNN_circuit_dynamic, tunn_down: {tunn_down}")

    num_tot_qubits = n_qubits

    n_qubits = n_qubits - num_ancillas

    print_cust(f"QCNN_circuit_dynamic, n_qubits: {n_qubits}, num_tot_qubits: {num_tot_qubits}")

    unpermuted_inputs = None

    num_layers = len(conv_params_tuple)

    print_cust(f"QCNN_circuit_dynamic, num_layers: {num_layers}")

    if num_layers > 0:
      print_cust(f"QCNN_circuit_dynamic, applying pennylane_angleencode at the start because num_layers > 0")
      pennylane_angleencode(range(0, n_qubits), inputs)
    wires = list(range(n_qubits))

    # TOMODIFY, layers: quick thing to tell the circuit whether or not I want HEA layers.
    is_hea = True
    # TOMODIFY, layers, HACK: hack to prespecify the types of layers as I want; please INJECT THIS IN.
    # layer_types_list = ["staircase", "staircase", "staircase", "staircase", "staircase"]

    if len(layer_types_list) == 0:
      # TOMODIFY, layers: some placeholder, just in case the layer types is not supplied.
      print_cust(f"QCNN_circuit_dynamic, len(layer_types_list) == 0, supplying some placeholder layer_types_list")
      layer_types_list = ["reversed_staircase" for _ in range(len(block_params_list))]
      print_cust(f"QCNN_circuit_dynamic, placeholder layer_types_list: {layer_types_list}")

    # Copy block_params_list so that each block is applied only once.
    block_params_remaining = block_params_list

    # Each layer contains: a block layer (if supplied), a convolutional layer, and a pooling layer.
    for layer in range(num_layers):
        print_cust(f"QCNN_circuit_dynamic, applying layer, layer: {layer}")
        # Check if any block parameter should be applied at this stage.
        new_block_params_remaining = []
        for bp in block_params_remaining:
            # bp.shape is (num_layers_bp, num_qubits_bp, 3)
            if bp.shape[1] == len(wires):
                num_layers_bp = bp.shape[0]
                block_variational_circuit(bp, num_layers_bp, wires)
            else:
                new_block_params_remaining.append(bp)
        block_params_remaining = new_block_params_remaining

        conv_params = conv_params_tuple[layer]
        pool_params = pool_params_tuple[layer]
        expansion_layer_data = expansion_data[layer]
        encoded_angles = None
        reversed_indices = None
        # NOTE: the encoded_angles here are INDICES of the input features, not the ACTUAL data values.
        # reversed_indices are the QUBIT LABELS/INDICES.
        if expansion_layer_data != []:
            encoded_angle_indices = expansion_layer_data[0]
            # relying on the fact that this is true, but it's OK i think
            if len(encoded_angle_indices) == len(inputs):
              unpermuted_inputs = unpermute_inputs(inputs, encoded_angle_indices)
              # print_cust(f"QCNN_circuit_dynamic, unpermuted_inputs: {unpermuted_inputs}")
            encoded_angles = [unpermuted_inputs[feat_idx] for feat_idx in encoded_angle_indices]
            reversed_indices = expansion_layer_data[1]
            # print_cust(f"QCNN_circuit_dynamic, encoded_angles: {encoded_angles}, reversed_indices: {reversed_indices}")

        # print_cust(f"QCNN_circuit_dynamic, encoded_angles: {encoded_angles}, reversed_indices: {reversed_indices}")

        # Apply a convolutional and pooling layer.
        conv_layer(conv_params, wires, encoded_angles=encoded_angles, reversed_indices=reversed_indices, pool_in=True)
        wires = pool_layer_with_measurement(pool_params, wires, pool_in=True)

    # Compute the number of output qubits needed.
    n_output_qubits = int(np.ceil(np.log2(n_classes)))
    if len(wires) < n_output_qubits:
        raise ValueError(f"Not enough wires left after pooling for the requested number of classes, len(wires): {len(wires)}, n_output_qubits: {n_output_qubits}.")

    # Apply another block layer on the last qubits.
    # NOTE: num_layers_applied is wrt to this specific variational block.
    num_layers_applied = 0
    # ret_probs_list_tunn = []
    for bp_idx, bp in enumerate(block_params_remaining):
      # TOMODIFY, layers: quick hack to see if ID init even helps.
      id_init_circ = False
      reupload_data = True
      print_cust(f"QCNN_circuit_dynamic, bp_idx: {bp_idx}, id_init_circ: {id_init_circ}")
      # bp.shape is (num_layers_bp, num_qubits_bp, 3)
      # NOTE, layers: if I'd like to have the smaller qubit count classifiers, I'd need to get rid of this below condition. (breaks
      # backwards compatibility, but whatever for now.)
      print_cust(f"QCNN_circuit_dynamic, len(wires): {len(wires)}")
      if bp.shape[1] == len(wires) or tunn_down:
          if (bp_idx % 2 == 0):
            axis = "Y"
          else:
            axis = "Y"
          print_cust(f"QCNN_circuit_dynamic, bp_idx: {bp_idx}, axis: {axis}")
          layer_type = layer_types_list[bp_idx]
          print_cust(f"QCNN_circuit_dynamic, layer_type: {layer_type}")
          if reupload_data:
            # HACKY, depthFL: using the size of the inputs
            if len(inputs.shape) == 2:
              print_cust(f"QCNN_circuit_dynamic, reupload_data, truncating input dim for inputs.shape == 2")
              reuploaded_inputs = inputs[:, -(len(wires)):]
            elif len(inputs.shape) == 1:
              print_cust(f"QCNN_circuit_dynamic, reupload_data, truncating input dim for inputs.shape == 1")
              reuploaded_inputs = inputs[-(len(wires)):]
            reupload_func = partial(pennylane_angleencode, qubits=wires, inputs=reuploaded_inputs, axis=axis, is_reuploading=True)
          else:
            reupload_func = None
          if id_init_circ:
            if len(inputs.shape) == 2:
              print_cust(f"QCNN_circuit_dynamic, id_init_circ, truncating input dim for inputs.shape == 2")
              reuploaded_inputs = inputs[:, -(len(wires)):]
            elif len(inputs.shape) == 1:
              print_cust(f"QCNN_circuit_dynamic, id_init_circ, truncating input dim for inputs.shape == 1")
              reuploaded_inputs = inputs[-(len(wires)):]
            inv_reupload_func = partial(pennylane_angleencode, qubits=wires, inputs=-1 * reuploaded_inputs, axis=axis, is_reuploading=True)
          else:
            inv_reupload_func = None
          print_cust(f"QCNN_circuit_dynamic, reupload_func: {reupload_func}, inv_reupload_func: {inv_reupload_func}")
          num_layers_bp = bp.shape[0]
          # TOMODIFY, layers, HACK: prespecify the order of the layer types.
          print_cust(f"QCNN_circuit_dynamic, bp_idx: {bp_idx}, num_layers_applied: {num_layers_applied}")
          block_variational_circuit(bp, num_layers_bp, wires, id_init_circ=id_init_circ, is_hea=is_hea, circuit_type=layer_type, reupload_func=reupload_func, inv_reupload_func=inv_reupload_func, offset_idx=num_layers_applied)
          # if id_init_circ:
          #   pennylane_angleencode(range(0, n_qubits), -1 * inputs, axis=axis, is_reuploading=True)
          num_layers_applied += num_layers_bp
          if num_ancillas > 0:
            print_cust(f"QCNN_circuit_dynamic, using ancillas for getting statistics")
            # TOMODIFY, depthFL: have a better way of extracting midcirc statistics; make the below line more
            # configurable.
            qml.CNOT(wires=[wires[0], n_qubits + bp_idx])
          if cheating:
            print_cust(f"QCNN_circuit_dynamic, bp_idx: {bp_idx}, applying cheating measurement on q0")
            qml.Snapshot(f"p0_bp{bp_idx}", measurement=qml.probs(wires=[0]))
          if tunn_down:
            print_cust(f"QCNN_circuit_dynamic, doing tunneling down accumulation")
            # ret_probs_list_tunn.append(qml.probs(wires=wires[0]))
            wires = wires[1:]

    # Instead of performing final pooling to a single wire, apply an extra convolution.
    # Here, we assume final_params is formatted like a conv layer
    # that acts on all the remaining wires.
    # Apply another convolution layer on the last qubits.
    if len(wires) == 1:
      qml.Rot(final_params[0, 0], final_params[0, 1], final_params[0, 2], wires=wires[0])
    else:
      if num_layers > 0:
        conv_layer(final_params, wires, pool_in=True)

    # Return the probability distribution for the first n_output_qubits.
    if num_ancillas == 0:
      if tunn_down:
        print_cust(f"QCNN_circuit_dynamic, returning list of probs from tunneling down")
        ret_probs_list_tunn = [qml.probs(wires=qubit_idx) for qubit_idx in range(0, len(block_params_remaining))]
        return ret_probs_list_tunn
      else:
        return qml.probs(wires=wires[:n_output_qubits])
    else:
      print_cust(f"QCNN_circuit_dynamic, returning ancilla statistics")
      n_ancillas_used = len(block_params_remaining)
      print_cust(f"QCNN_circuit_dynamic, n_ancillas_used: {n_ancillas_used}")
      return [qml.probs(wires=ancilla_qubit_idx) for ancilla_qubit_idx in range(n_qubits, n_qubits + n_ancillas_used)]