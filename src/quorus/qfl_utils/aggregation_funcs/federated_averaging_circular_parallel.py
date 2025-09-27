import torch
from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
from quorus.qfl_utils.aggregation_funcs.agg_utils.agg_disc_params import aggregate_discriminator_params
from quorus.quantum_circuit_funcs.utils.qcnn_code import compute_conv_layers
from quorus.qfl_utils.aggregation_funcs.agg_utils.block_params_agg import aggregate_block_params_general

"""#### Parallel aggregation function"""

def federated_averaging_circular_parallel(client_params_dict, clients_data_dict, n_output_qubits=1, generative=False, is_qcnn=True):
    """
    Perform federated averaging in a chunkwise (parallel) manner.

    This function aggregates parameters from heterogeneous clients.
    Each client's QCNN has the structure:
      - Angle encoding on n qubits
      - An n-qubit HEA ansatz (block ansatz) with a specified number of layers
      - An n-qubit convolutional layer
      - An n-qubit pooling layer (reducing to n/2 qubits)
      - n/2-qubit HEA ansatz with specified number of layers
      - n/2-qubit convolutional layer, etc.

    For the block ansatz parameters (stored in params[5]), different client types can have
    different numbers of layers. The common (shared) block parameters are defined as the last L_common
    layers, where L_common is the minimum block depth across all client types (that have block parameters).
    Extra block layers (i.e. those not in the common group) are aggregated separately for each client type.

    Aggregation is performed via weighted circular averaging, where weights are the number of training samples.

    Args:
      client_params_dict: dict mapping client types (e.g., model sizes) to lists of parameter tuples.
          Each tuple is (conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list).
      clients_data_dict: dict mapping client types to lists of client data tuples.
          Each client data tuple is ((X_train, y_train), (X_val, y_val)).
      n_output_qubits: an integer representing the number of output qubits for the QCNN model.

    Returns:
      aggregated_params: a meta aggregated parameter tuple with structure:
         (
           aggregated_conv_params,  # tuple of meta conv layers (even, odd parts)
           aggregated_pool_params,  # tuple of meta pool layers
           aggregated_final_pool_param,
           aggregated_final_params,
           aggregated_bias,
           (aggregated_common_block, aggregated_extra_block)
         )
         where aggregated_common_block is a tuple of L_common arrays (the shared block layers)
         and aggregated_extra_block is a dict mapping client type to a tuple of aggregated extra block layers.
    """

    # --- Aggregate convolution, pooling, final pooling, final rotation, and bias ---
    # NOTE, layers: if not is_qcnn, then set math_int to be torch.
    # ^ might be buggy if I decide to use this function. only numpy is supported for non-block params, but
    # for block params, torch is supported.
    if generative or not is_qcnn:
      math_int = torch
    else:
      math_int = np

    print_cust(f"federated_averaging_circular_parallel, math_int: {math_int}")

    # Determine meta number of convolution layers based on the largest client (assumes model size key represents qubit count)
    max_model_size = max(client_params_dict.keys())
    # meta_conv_layers = int(np.log2(max_model_size))
    meta_conv_layers = compute_conv_layers(max_model_size, n_output_qubits, generative=generative, is_qcnn=is_qcnn)

    # Use a reference from a client of the max model size
    ref_params = client_params_dict[max_model_size][0]
    ref_conv = ref_params[0]

    # Initialize accumulators for conv parameters (even and odd parts)
    accum_conv_even = []
    accum_conv_odd = []
    total_weight_conv = [0.0] * meta_conv_layers
    for l in range(meta_conv_layers):
        even_shape = ref_conv[l][0].shape
        odd_shape = ref_conv[l][1].shape
        accum_conv_even.append(np.zeros(even_shape, dtype=np.complex128))
        accum_conv_odd.append(np.zeros(odd_shape, dtype=np.complex128))

    # Initialize accumulators for pooling parameters
    accum_pool = []
    total_weight_pool = [0.0] * meta_conv_layers
    ref_pool = ref_params[1]
    for l in range(meta_conv_layers):
        pool_shape = ref_pool[l].shape
        accum_pool.append(np.zeros(pool_shape, dtype=np.complex128))

    # Accumulators for final pooling, final rotation, and bias.
    accum_final_pool = 0 + 0j
    if math_int == np:
      accum_final_params = math_int.zeros(ref_params[3].shape, dtype=np.complex128)
      accum_bias = math_int.zeros(ref_params[4].shape, dtype=np.float64)
    else:
      accum_final_params = math_int.zeros(ref_params[3].shape, dtype=torch.complex64)
      accum_bias = math_int.zeros(ref_params[4].shape, dtype=torch.float32)
    total_weight_final = 0.0

    # Loop over all clients to accumulate parameters
    for client_type in client_params_dict:
        for params, data in zip(client_params_dict[client_type], clients_data_dict[client_type]):
            train_data, _ = data
            weight = len(train_data[0])

            # Convolution parameters: map client's conv layers into meta layers
            conv_params = params[0]
            client_conv_layers = len(conv_params)
            offset = meta_conv_layers - client_conv_layers
            for l in range(client_conv_layers):
                meta_layer = l + offset
                even_angles = conv_params[l][0]
                odd_angles  = conv_params[l][1]
                accum_conv_even[meta_layer] += weight * np.exp(1j * even_angles)
                accum_conv_odd[meta_layer]  += weight * np.exp(1j * odd_angles)
                total_weight_conv[meta_layer] += weight

            # Pooling parameters: similar mapping
            client_pool = params[1]
            client_pool_layers = len(client_pool)
            offset_pool = meta_conv_layers - client_pool_layers
            for l in range(client_pool_layers):
                meta_layer = l + offset_pool
                pool_angles = client_pool[l]
                accum_pool[meta_layer] += weight * np.exp(1j * pool_angles)
                total_weight_pool[meta_layer] += weight

            # Final pooling, rotation, and bias
            final_pool_param = params[2]
            accum_final_pool += weight * math_int.exp(1j * final_pool_param[0])
            final_params = params[3]
            accum_final_params += weight * math_int.exp(1j * final_params)
            bias_param = params[4]
            accum_bias += weight * bias_param
            total_weight_final += weight

    meta_avg_conv = []
    for l in range(meta_conv_layers):
        avg_even = np.angle(accum_conv_even[l] / total_weight_conv[l])
        avg_odd  = np.angle(accum_conv_odd[l] / total_weight_conv[l])
        meta_avg_conv.append((avg_even, avg_odd))
    meta_avg_conv = tuple(meta_avg_conv)

    meta_avg_pool = []
    for l in range(meta_conv_layers):
        avg_pool = np.angle(accum_pool[l] / total_weight_pool[l])
        meta_avg_pool.append(avg_pool)
    meta_avg_pool = tuple(meta_avg_pool)

    if math_int == np:
      avg_final_pool_param = math_int.array([math_int.angle(accum_final_pool / total_weight_final)])
      avg_final_params = math_int.angle(accum_final_params / total_weight_final)
    else:
      avg_final_pool_param = math_int.tensor([math_int.angle(accum_final_pool / total_weight_final)])
      avg_final_params = math_int.angle(accum_final_params / total_weight_final)
    avg_bias = accum_bias / total_weight_final

    # NOTE: in the generative case (I do instanceof checks to check), this aggregates the generator params only.
    # forcing use of torch here. can change later
    with torch.no_grad():
      aggregated_block_params = aggregate_block_params_general(client_params_dict, clients_data_dict, math_int=math_int)
      if generative:
        # assumed, if generative, that (1) block_params contains discriminator, and (2) there exists at least one discriminator.
        # ALSO, this assumes that if a client is in the training, then its disc params need to be aggregated.
        agg_disc = aggregate_discriminator_params(client_params_dict, clients_data_dict)
        # this changes how the block params need to be used downstream dep on generative... not sure how else to adapt it otherwise for now
        aggregated_block_params = [aggregated_block_params, agg_disc]

    # --- Build final aggregated parameters tuple ---
    aggregated_params = (
        meta_avg_conv,
        meta_avg_pool,
        avg_final_pool_param,
        avg_final_params,
        avg_bias,
        aggregated_block_params
    )
    return aggregated_params