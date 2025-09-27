from pennylane import numpy as np

"""### Broadcast parameters function, aggregated shared only, no convolutional layer aggregation"""

def broadcast_param_updates_shared_noconv(client_params_dict, aggregated_params):
    """
    Update each client's parameters in client_params_dict using only the shared (common) aggregated parameters.

    Args:
      client_params_dict: dict mapping client types to lists of parameter tuples. Each parameter tuple has the form:
          (conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list)
          where:
            conv_params: tuple of convolution parameters (each element is a tuple (even, odd))
            pool_params: tuple of pooling parameters (arrays)
            final_pool_param: array for final pooling
            final_params: array for final rotation
            bias_param: array (or scalar) for bias
            block_params_list: list of block (HEA ansatz) parameter arrays.
      aggregated_params: aggregated parameters returned by
          federated_averaging_circular_parallel_shared, having the structure:
          (shared_conv_avg, shared_pool_avg, shared_final_pool_param,
           shared_final_params, shared_bias, shared_block_params)

    Returns:
      Updated client_params_dict with only the shared (common) portions replaced by the aggregated ones.
    """
    # NOTE, layers: I shouldn't need to use this function for the layer-heterogenous case, so I'm not going to
    # modify it for now.

    # Unpack aggregated shared parameters.
    shared_conv_avg, shared_pool_avg, shared_final_pool_param, shared_final_params, shared_bias, shared_block_params = aggregated_params

    # Number of shared convolution and pooling layers (these are the "common" layers to update).
    num_shared_conv = len(shared_conv_avg)
    num_shared_pool = len(shared_pool_avg)

    # Loop over each client type and each client.
    for client_type in client_params_dict:
        for i, params in enumerate(client_params_dict[client_type]):
            conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list = params

            # --- Update QCNN convolution parameters ---
            # conv_params is a tuple; update only the last 'num_shared_conv' layers.
            client_conv_layers = len(conv_params)
            offset_conv = client_conv_layers - num_shared_conv
            # new_conv_params = list(conv_params)  # convert to list for modification
            # for j in range(num_shared_conv):
            #     # Each conv layer is a tuple (even, odd) and we simply replace it.
            #     new_conv_params[offset_conv + j] = shared_conv_avg[j]
            new_conv_params = conv_params

            # --- Update pooling parameters ---
            client_pool_layers = len(pool_params)
            offset_pool = client_pool_layers - num_shared_pool
            new_pool_params = list(pool_params)
            for j in range(num_shared_pool):
                new_pool_params[offset_pool + j] = shared_pool_avg[j]

            # --- Update final pooling, rotation, and bias parameters ---
            # These parameters are assumed to be shared among all clients.
            new_final_pool_param = np.array(shared_final_pool_param)
            if isinstance(shared_final_params, tuple):
              new_final_params = tuple(shared_final_params)
            else:
              new_final_params = np.array(shared_final_params)
            new_bias_param = np.array(shared_bias) if isinstance(shared_bias, np.ndarray) else shared_bias

            # --- Update block (HEA ansatz) parameters ---
            # For each block parameter array in the client's list, update only the last common layers.
            new_block_params_list = []
            for bp in block_params_list:
                # Determine the block type by number of qubits (assumed bp.shape = (depth, n_qubits, 3)).
                block_type = bp.shape[1]
                if block_type in shared_block_params:
                    aggregated_common = shared_block_params[block_type]  # shape: (common_depth, block_type, 3)
                    common_depth = aggregated_common.shape[0]
                    client_depth = bp.shape[0]
                    bp_updated = bp.copy()
                    # Update the last 'common_depth' layers.
                    bp_updated[client_depth - common_depth : client_depth, :, :] = aggregated_common
                    new_block_params_list.append(bp_updated)
                else:
                    # If no update for this block type is provided, leave it unchanged.
                    new_block_params_list.append(bp)

            # Assemble the new parameter tuple for the client.
            updated_params = (
                tuple(new_conv_params),
                tuple(new_pool_params),
                new_final_pool_param,
                new_final_params,
                new_bias_param,
                new_block_params_list
            )
            # Update the client parameter in place.
            client_params_dict[client_type][i] = updated_params

    return client_params_dict