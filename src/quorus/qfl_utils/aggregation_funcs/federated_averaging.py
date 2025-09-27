from pennylane import numpy as np

"""## Federated averaging function

### Arithmetic mean
"""

def federated_averaging(client_params_dict, clients_data_dict):
    # Get a reference parameter structure.
    first_client_params = None
    for client_type in client_params_dict:
        if len(client_params_dict[client_type]) > 0:
            first_client_params = client_params_dict[client_type][0]
            break
    if first_client_params is None:
        raise ValueError("No client parameters provided.")
    conv_params_first, pool_params_first, final_pool_param_first, final_params_first, bias_param_first, block_params_first = first_client_params

    conv_layers = len(conv_params_first)
    aggregated_conv_params = []
    for layer in range(conv_layers):
        even_shape = conv_params_first[layer][0].shape
        odd_shape = conv_params_first[layer][1].shape
        aggregated_even = np.zeros(even_shape)
        aggregated_odd = np.zeros(odd_shape)
        aggregated_conv_params.append((aggregated_even, aggregated_odd))
    pool_layers = len(pool_params_first)
    aggregated_pool_params = [np.zeros(pool_params_first[layer].shape) for layer in range(pool_layers)]
    aggregated_final_pool_param = np.zeros(final_pool_param_first.shape)
    aggregated_final_params = np.zeros(final_params_first.shape)
    aggregated_bias_param = np.zeros(bias_param_first.shape)
    # Initialize accumulator for block parameters.
    aggregated_block_params = []
    for bp in block_params_first:
        aggregated_block_params.append(np.zeros(bp.shape))

    total_weight = 0.0
    for client_type in client_params_dict:
        clients_params = client_params_dict[client_type]
        clients_data = clients_data_dict[client_type]
        for params, data in zip(clients_params, clients_data):
            train_data, _ = data
            X_train, _ = train_data
            weight = len(X_train)
            total_weight += weight
            conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list = params
            for layer in range(conv_layers):
                aggregated_conv_params[layer] = (
                    aggregated_conv_params[layer][0] + weight * conv_params[layer][0],
                    aggregated_conv_params[layer][1] + weight * conv_params[layer][1]
                )
            for layer in range(len(pool_params)):
                aggregated_pool_params[layer] += weight * pool_params[layer]
            aggregated_final_pool_param += weight * final_pool_param
            aggregated_final_params += weight * final_params
            aggregated_bias_param += weight * bias_param
            # Aggregate block parameters.
            for i, bp in enumerate(block_params_list):
                aggregated_block_params[i] += weight * bp

    for layer in range(conv_layers):
        aggregated_conv_params[layer] = (
            aggregated_conv_params[layer][0] / total_weight,
            aggregated_conv_params[layer][1] / total_weight
        )
    aggregated_pool_params = [pp / total_weight for pp in aggregated_pool_params]
    aggregated_final_pool_param /= total_weight
    aggregated_final_params /= total_weight
    aggregated_bias_param /= total_weight
    for i in range(len(aggregated_block_params)):
        aggregated_block_params[i] /= total_weight

    aggregated_params = (
        tuple(aggregated_conv_params),
        tuple(aggregated_pool_params),
        aggregated_final_pool_param,
        aggregated_final_params,
        aggregated_bias_param,
        aggregated_block_params
    )
    return aggregated_params