from pennylane import numpy as np

"""### Circular mean"""

def federated_averaging_circular(client_params_dict, clients_data_dict):
    """
    Federated averaging using circular averaging for angle parameters,
    including the block_params_list.
    For each parameter (including each rotation gate’s angles in block parameters),
    we compute the weighted circular mean:
         avg_angle = angle(sum(weight * exp(1j*angle))).
    """
    # Get a reference parameter structure.
    first_client_params = None
    for client_type in client_params_dict:
        if len(client_params_dict[client_type]) > 0:
            first_client_params = client_params_dict[client_type][0]
            break
    if first_client_params is None:
        raise ValueError("No client parameters provided.")
    conv_params_tuple_ref, pool_params_tuple_ref, final_pool_param_ref, final_params_ref, bias_param_ref, block_params_ref = first_client_params
    n_layers = len(conv_params_tuple_ref)

    # --- Accumulators for convolution parameters (circular) ---
    accum_conv_even = []
    accum_conv_odd = []
    for layer in range(n_layers):
        even_shape = conv_params_tuple_ref[layer][0].shape
        odd_shape  = conv_params_tuple_ref[layer][1].shape
        accum_conv_even.append(np.zeros(even_shape, dtype=np.complex128))
        accum_conv_odd.append(np.zeros(odd_shape, dtype=np.complex128))

    # --- Accumulators for pooling, final rotation, and bias ---
    accum_pool = []
    for layer in range(n_layers):
        pool_shape = pool_params_tuple_ref[layer].shape
        accum_pool.append(np.zeros(pool_shape, dtype=np.complex128))
    accum_final_pool = 0 + 0j
    accum_final_params = np.zeros(final_params_ref.shape, dtype=np.complex128)
    accum_bias = np.zeros(bias_param_ref.shape, dtype=np.float64)

    # --- New: Accumulators for block parameters ---
    # For each block parameter array (shape (L_bp, n_bp, 3)), accumulate using exp(1j * angle) elementwise.
    accum_block = []
    for bp in block_params_ref:
        accum_block.append(np.zeros(bp.shape, dtype=np.complex128))

    total_weight = 0.0

    # --- Loop over clients ---
    for client_type in client_params_dict:
        clients_params = client_params_dict[client_type]
        clients_data = clients_data_dict[client_type]
        for params, data in zip(clients_params, clients_data):
            train_data, _ = data
            X_train, _ = train_data
            weight = len(X_train)
            total_weight += weight
            conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list = params

            # Aggregate convolution parameters.
            for layer in range(n_layers):
                even_params = conv_params[layer][0]
                odd_params  = conv_params[layer][1]
                accum_conv_even[layer] += weight * np.exp(1j * even_params)
                accum_conv_odd[layer]  += weight * np.exp(1j * odd_params)

            # Aggregate pooling parameters.
            for layer in range(n_layers):
                accum_pool[layer] += weight * np.exp(1j * pool_params[layer])

            # Aggregate final pooling parameter.
            accum_final_pool += weight * np.exp(1j * final_pool_param[0])

            # Aggregate final rotation parameters.
            accum_final_params += weight * np.exp(1j * final_params)

            # Aggregate bias (arithmetic).
            accum_bias += weight * bias_param

            # --- Aggregate block parameters ---
            for idx, bp in enumerate(block_params_list):
                accum_block[idx] += weight * np.exp(1j * bp)

    # --- Finalize convolution parameters ---
    avg_conv_layers = []
    for layer in range(n_layers):
        avg_even = np.angle(accum_conv_even[layer])
        avg_odd  = np.angle(accum_conv_odd[layer])
        avg_conv_layers.append((avg_even, avg_odd))

    # --- Finalize pooling parameters ---
    avg_pool_layers = []
    for layer in range(n_layers):
        avg_pool = np.angle(accum_pool[layer])
        avg_pool_layers.append(avg_pool)

    # --- Finalize final pooling and rotation ---
    avg_final_pool_param = np.array([np.angle(accum_final_pool)])
    avg_final_params = np.angle(accum_final_params)

    # --- Finalize bias ---
    avg_bias = accum_bias / total_weight

    # --- Finalize block parameters ---
    avg_block_params = []
    for idx, bp in enumerate(block_params_ref):
        avg_bp = np.angle(accum_block[idx])
        avg_block_params.append(avg_bp)

    aggregated_params = (
        tuple(avg_conv_layers),
        tuple(avg_pool_layers),
        avg_final_pool_param,
        avg_final_params,
        avg_bias,
        avg_block_params
    )
    return aggregated_params