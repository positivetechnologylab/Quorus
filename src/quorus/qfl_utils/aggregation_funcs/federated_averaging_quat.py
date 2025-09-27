from quorus.qfl_utils.aggregation_funcs.agg_utils.quaternion_agg_funcs import euler_to_quat, quat_to_euler
from pennylane import numpy as np

def federated_averaging_quat(client_params_dict, clients_data_dict):
    """
    Federated averaging using weighted quaternion averaging for rotations,
    including aggregation of block_params_list.
    For each rotation (specified as Euler angles) in the convolutional layers
    and in each block parameter array (shape (L_bp, n_bp, 3)), we convert to a quaternion,
    accumulate a weighted sum, then convert the averaged quaternion back to Euler angles.
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
    num_layers = len(conv_params_tuple_ref)

    # --- Accumulators for convolution parameters (quaternion averaging) ---
    accum_conv_even = []
    accum_conv_odd = []
    for layer in range(num_layers):
        even_shape = conv_params_tuple_ref[layer][0].shape  # (N_even, 12)
        odd_shape  = conv_params_tuple_ref[layer][1].shape   # (N_odd, 12)
        # There are 4 rotations per row (each 3 parameters).
        accum_conv_even.append(np.zeros((even_shape[0], 4, 4)))
        accum_conv_odd.append(np.zeros((odd_shape[0], 4, 4)))

    # --- Accumulators for pooling, final rotation and bias ---
    accum_pool = []
    for layer in range(num_layers):
        pool_shape = pool_params_tuple_ref[layer].shape  # (N_pool, 1)
        accum_pool.append(np.zeros(pool_shape, dtype=np.complex128))
    accum_final = np.zeros(4, dtype=np.float64)
    accum_final_pool = 0 + 0j
    accum_bias = np.zeros(bias_param_ref.shape, dtype=np.float64)

    # --- New: Accumulators for block parameters ---
    # For each block parameter array (shape (L_bp, n_bp, 3)), create an accumulator (shape (L_bp, n_bp, 4)).
    accum_block = []
    for bp in block_params_ref:
        shape = bp.shape
        accum_block.append(np.zeros((shape[0], shape[1], 4)))

    total_weight = 0.0

    # --- Loop over clients and accumulate weighted sums ---
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
            for layer in range(num_layers):
                even_params = conv_params[layer][0]  # shape (N_even, 12)
                odd_params  = conv_params[layer][1]   # shape (N_odd, 12)
                for i in range(even_params.shape[0]):
                    for r in range(4):
                        euler_angles = even_params[i, r*3:(r+1)*3]
                        q = euler_to_quat(euler_angles[0], euler_angles[1], euler_angles[2])
                        if np.linalg.norm(accum_conv_even[layer][i, r]) > 0:
                            if np.dot(accum_conv_even[layer][i, r], q) < 0:
                                q = -q
                        accum_conv_even[layer][i, r] += weight * q
                for i in range(odd_params.shape[0]):
                    for r in range(4):
                        euler_angles = odd_params[i, r*3:(r+1)*3]
                        q = euler_to_quat(euler_angles[0], euler_angles[1], euler_angles[2])
                        if np.linalg.norm(accum_conv_odd[layer][i, r]) > 0:
                            if np.dot(accum_conv_odd[layer][i, r], q) < 0:
                                q = -q
                        accum_conv_odd[layer][i, r] += weight * q

            # Aggregate pooling parameters.
            for layer in range(num_layers):
                pool_layer_params = pool_params[layer]  # shape (N_pool, 1)
                for i in range(pool_layer_params.shape[0]):
                    angle = pool_layer_params[i, 0]
                    accum_pool[layer][i, 0] += weight * np.exp(1j * angle)

            # Aggregate final rotation parameters.
            euler_angles = final_params[0]  # (phi, theta, omega)
            q = euler_to_quat(euler_angles[0], euler_angles[1], euler_angles[2])
            if np.linalg.norm(accum_final) > 0:
                if np.dot(accum_final, q) < 0:
                    q = -q
            accum_final += weight * q

            # Aggregate final pooling parameter.
            angle = final_pool_param[0]
            accum_final_pool += weight * np.exp(1j * angle)

            # Aggregate bias (arithmetic).
            accum_bias += weight * bias_param

            # --- Aggregate block parameters ---
            # For each block parameter array in the list.
            for idx, bp in enumerate(block_params_list):
                shape = bp.shape  # (L_bp, n_bp, 3)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        euler_angles = bp[i, j, :]
                        q = euler_to_quat(euler_angles[0], euler_angles[1], euler_angles[2])
                        if np.linalg.norm(accum_block[idx][i, j]) > 0:
                            if np.dot(accum_block[idx][i, j], q) < 0:
                                q = -q
                        accum_block[idx][i, j] += weight * q

    # --- Finalize averages for convolution parameters ---
    avg_conv_layers = []
    for layer in range(num_layers):
        even_shape = conv_params_tuple_ref[layer][0].shape
        odd_shape = conv_params_tuple_ref[layer][1].shape
        avg_even = np.zeros(even_shape)
        avg_odd = np.zeros(odd_shape)
        for i in range(even_shape[0]):
            for r in range(4):
                q_avg = accum_conv_even[layer][i, r] / np.linalg.norm(accum_conv_even[layer][i, r])
                euler_avg = quat_to_euler(q_avg)
                avg_even[i, r*3:(r+1)*3] = euler_avg
        for i in range(odd_shape[0]):
            for r in range(4):
                q_avg = accum_conv_odd[layer][i, r] / np.linalg.norm(accum_conv_odd[layer][i, r])
                euler_avg = quat_to_euler(q_avg)
                avg_odd[i, r*3:(r+1)*3] = euler_avg
        avg_conv_layers.append((avg_even, avg_odd))

    # --- Finalize pooling parameters ---
    avg_pool_layers = []
    for layer in range(num_layers):
        pool_shape = pool_params_tuple_ref[layer].shape
        avg_pool = np.zeros(pool_shape)
        for i in range(pool_shape[0]):
            avg_pool[i, 0] = np.angle(accum_pool[layer][i, 0])
        avg_pool_layers.append(avg_pool)

    # --- Finalize final rotation ---
    q_final_avg = accum_final / np.linalg.norm(accum_final)
    avg_final_params = np.array([quat_to_euler(q_final_avg)])

    # --- Finalize final pooling ---
    avg_final_pool_param = np.array([np.angle(accum_final_pool)])

    # --- Finalize bias ---
    avg_bias = accum_bias / total_weight

    # --- Finalize block parameters ---
    avg_block_params = []
    for idx, bp in enumerate(block_params_ref):
        shape = bp.shape
        avg_bp = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                q_avg = accum_block[idx][i, j] / np.linalg.norm(accum_block[idx][i, j])
                euler_avg = quat_to_euler(q_avg)
                avg_bp[i, j, :] = euler_avg
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