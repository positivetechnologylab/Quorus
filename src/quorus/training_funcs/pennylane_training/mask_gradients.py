from pennylane import numpy as np

"""## Gradient masking function"""

def mask_gradients(params):
    """
    For a given set of parameters, creates a gradient mask for the set of convolutional layers, pooling layers, and block parameters acting on the initial set of qubits
    (i.e., creates a Boolean mask that is 1 for the first convolution, pooling, and block layers, and 0 everywhere else).

    Parameters:
      params: a list of arrays of parameters for this client

    Returns:
      A list of arrays representing a boolean mask for the gradient for this client such that only parameters on all of its qubits are updated.
    """
    conv_params_first, pool_params_first, final_pool_param_first, final_params_first, bias_param_first, block_params_first = params
    conv_layers = len(conv_params_first)
    aggregated_conv_mask = []
    for layer in range(conv_layers):
        even_shape = conv_params_first[layer][0].shape
        odd_shape = conv_params_first[layer][1].shape
        if layer == 0:
            aggregated_even = np.ones(even_shape)
            aggregated_odd = np.ones(odd_shape)
        else:
            aggregated_even = np.zeros(even_shape)
            aggregated_odd = np.zeros(odd_shape)
        aggregated_conv_mask.append((aggregated_even, aggregated_odd))
    pool_layers = len(pool_params_first)
    aggregated_pool_mask = []
    for layer in range(pool_layers):
        aggregated_pool_mask.append(np.ones(pool_params_first[layer].shape) if layer == 0 else np.zeros(pool_params_first[layer].shape))
    aggregated_final_pool_mask = np.zeros(final_pool_param_first.shape)
    aggregated_final_params_mask = np.zeros(final_params_first.shape)
    aggregated_bias_mask = np.zeros(bias_param_first.shape)

    # Determine the number of qubits in the first convolutional layer.
    # We approximate n_first = 2 * (number of even pairs) from conv_params_first[0].
    n_first = conv_params_first[0][0].shape[0] * 2
    aggregated_block_mask = []
    for bp in block_params_first:
        # bp.shape = (num_layers_bp, num_qubits_bp, 3)
        if bp.shape[1] == n_first:
            mask_bp = np.ones(bp.shape)
        else:
            mask_bp = np.zeros(bp.shape)
        aggregated_block_mask.append(mask_bp)

    aggregated_params = (
        tuple(aggregated_conv_mask),
        tuple(aggregated_pool_mask),
        aggregated_final_pool_mask,
        aggregated_final_params_mask,
        aggregated_bias_mask,
        aggregated_block_mask
    )
    return aggregated_params