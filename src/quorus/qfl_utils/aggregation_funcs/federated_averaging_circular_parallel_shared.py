from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
import torch
from quorus.qfl_utils.aggregation_funcs.agg_utils.agg_disc_params import aggregate_discriminator_params
from quorus.qfl_utils.aggregation_funcs.agg_utils.block_params_agg import aggregate_block_params_general

"""#### Parallel aggregation function, shared only"""

# TODO: edit fed_avg_circ_par_shared, just like I did for fed_avg_circ_parallel.

def federated_averaging_circular_parallel_shared(client_params_dict, clients_data_dict, generative=False):
    """
    Perform federated averaging only over those parameters that are common to all clients.

    In our dynamic QCNN setup the client parameters tuple is assumed to have the structure:
      (conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list)

    where:
      - conv_params is a tuple of convolution layer parameters (each element is a tuple (even, odd)
        whose arrays hold rotation angles used in the QCNN).
      - pool_params is a tuple of pooling layer parameters.
      - final_pool_param, final_params, and bias_param are the parameters used after all convolutions.
      - block_params_list is a list of arrays corresponding to the block (HEA ansatz) parameters.

    For heterogeneous clients (e.g. some with 4 qubits and others with 8 qubits) only the QCNN and block
    layers that every client shares are aggregated. For example, if the 4-qubit clients have 2 QCNN layers
    and the 8-qubit clients have 3, then only the final two layers are common.

    The aggregation is done using weighted circular averaging: for any parameter (which is an angle),
    we compute the weighted sum of exp(1j*angle) and then take np.angle(sum/total_weight).

    For block parameters, the function calls aggregate_block_params_general (which groups by block type
    and aggregates both “common” and “extra” parts) and then it keeps only the common part.

    Args:
      client_params_dict: dict mapping client types (e.g. model sizes) to lists of parameter tuples.
      clients_data_dict: dict mapping client types to the corresponding list of data tuples.
         (Each data tuple is ((X_train, y_train), (X_val, y_val)) and the length of X_train is used as weight.)

    Returns:
      A tuple of aggregated parameters for the shared structure:
         (shared_conv_params, shared_pool_params, shared_final_pool_param,
          shared_final_params, shared_bias, shared_block_params)
      where:
         - shared_conv_params is a tuple (length = min_conv_layers) of (even, odd) parameters.
         - shared_pool_params is a tuple (length = min_conv_layers).
         - shared_final_pool_param, shared_final_params, and shared_bias are the aggregated final parameters.
         - shared_block_params is a dict (by block type) where each value is the aggregated common block parameters.
    """

    print_cust(f"federated_averaging_circular_parallel_shared, generative: {generative}")

    # 1. Determine the minimum number of convolution (QCNN) layers across all clients.
    min_conv_layers = None
    for client_type in client_params_dict:
        for params in client_params_dict[client_type]:
            client_conv_layers = len(params[0])
            if min_conv_layers is None or client_conv_layers < min_conv_layers:
                min_conv_layers = client_conv_layers
    if min_conv_layers is None and not generative:
      # NOTE, layers: am changing the logic to allow for NO convolutional and ONLY block params.
      min_conv_layers = 0
      print_cust(f"federated_averaging_circular_parallel_shared, min_conv_layers is None and not generative, min_conv_layers: {min_conv_layers}")
    elif min_conv_layers is None and generative:
      min_conv_layers = 0
      print_cust(f"federated_averaging_circular_parallel_shared, min_conv_layers is None and generative, min_conv_layers: {min_conv_layers}")

    # NOTE, layers (and should prob change): min_conv_layers being 0 is a "hack" for saying we are in the
    # non-QCNN regime, and thus doing model training in PyTorch (although these things should not need)
    # to be coupled with one another.
    # NOTE, layers (and should prob change): all these things regarding torch interfaces, dtypes, etc. should
    # be injected.
    math_int = np
    complex_dtype = np.complex128
    float_dtype = np.float64
    container_func = np.array

    print_cust(f"federated_averaging_circular_parallel_shared, math_int: {math_int}, complex_dtype: {complex_dtype}, float_dtype: {float_dtype}")

    print_cust(f"federated_averaging_circular_parallel_shared, min_conv_layers: {min_conv_layers}")

    # 2. Initialize accumulators for the shared QCNN layers (even and odd parts) and pooling layers.
    shared_conv_even_accum = [None] * min_conv_layers
    shared_conv_odd_accum = [None] * min_conv_layers
    total_weight_conv = [0.0] * min_conv_layers

    shared_pool_accum = [None] * min_conv_layers
    total_weight_pool = [0.0] * min_conv_layers

    # Use a reference client that has exactly min_conv_layers layers to initialize our accumulators.
    ref_found = False
    for client_type in client_params_dict:
        for params in client_params_dict[client_type]:
            client_conv = params[0]
            if len(client_conv) == min_conv_layers:
                for i in range(min_conv_layers):
                    shared_conv_even_accum[i] = math_int.zeros(client_conv[i][0].shape, dtype=complex_dtype)
                    shared_conv_odd_accum[i] = math_int.zeros(client_conv[i][1].shape, dtype=complex_dtype)
                    shared_pool_accum[i] = math_int.zeros(params[1][i].shape, dtype=complex_dtype)
                ref_found = True
                break
        if ref_found:
            break

    # 3. Initialize accumulators for final pooling, final rotation, and bias.
    any_client = next(iter(next(iter(client_params_dict.values()))))
    accum_final_pool = 0 + 0j
    if isinstance(any_client[3], (list, tuple)):
      accum_final_params = [math_int.zeros(any_client[3][0].shape, dtype=complex_dtype), math_int.zeros(any_client[3][1].shape, dtype=complex_dtype)]
    else:
      accum_final_params = math_int.zeros(any_client[3].shape, dtype=complex_dtype)
    accum_bias = math_int.zeros(any_client[4].shape, dtype=float_dtype)
    total_weight_final = 0.0
    print_cust(f"federated_averaging_circular_parallel_shared, accum_final_params: {accum_final_params}")
    if isinstance(accum_final_params, (list, tuple)):
      print_cust(f"federated_averaging_circular_parallel_shared, accum_final_params[0]: {accum_final_params[0]}, accum_final_params[1]: {accum_final_params[1]}, type(accum_final_params[0]): {type(accum_final_params[0])}, type(accum_final_params[1]): {type(accum_final_params[1])}")

    # 4. Loop over all clients and accumulate only the shared parts.
    # For each client we extract the last min_conv_layers layers.
    for client_type in client_params_dict:
        print_cust(f"federated_averaging_circular_parallel_shared, client_type: {client_type}")
        for params, data in zip(client_params_dict[client_type], clients_data_dict[client_type]):
            # Use the number of training samples as the weight.
            train_data = data[0]
            weight = len(train_data[0])
            print_cust(f"federated_averaging_circular_parallel_shared, weight: {weight}")
            client_conv = params[0]
            client_pool = params[1]
            client_layers = len(client_conv)
            offset_conv = client_layers - min_conv_layers
            for i in range(min_conv_layers):
                conv_layer = client_conv[offset_conv + i]
                even_angles = conv_layer[0]
                odd_angles = conv_layer[1]
                shared_conv_even_accum[i] += weight * math_int.exp(1j * even_angles)
                shared_conv_odd_accum[i] += weight * math_int.exp(1j * odd_angles)
                total_weight_conv[i] += weight
            client_pool_layers = len(client_pool)
            offset_pool = client_pool_layers - min_conv_layers
            for i in range(min_conv_layers):
                pool_layer = client_pool[offset_pool + i]
                shared_pool_accum[i] += weight * math_int.exp(1j * pool_layer)
                total_weight_pool[i] += weight

            # Accumulate final pooling, rotation parameters, and bias.
            final_pool_param = params[2]
            accum_final_pool += weight * math_int.exp(1j * final_pool_param[0])
            final_params = params[3]
            print_cust(f"federated_averaging_circular_parallel_shared, any_client[3]: {any_client[3]}")
            if isinstance(any_client[3], (list, tuple)):
              print_cust(f"federated_averaging_circular_parallel_shared, final_params: {final_params}")
              print_cust(f"federated_averaging_circular_parallel_shared, type(final_params[0]): {type(final_params[0])}, type(final_params[1]): {type(final_params[1])}")
              accum_final_params[0] += weight * math_int.exp(1j * final_params[0])
              accum_final_params[1] += weight * math_int.exp(1j * final_params[1])
              print_cust(f"federated_averaging_circular_parallel_shared, type(accum_final_params[0]): {type(accum_final_params[0])}, type(accum_final_params[1]): {type(accum_final_params[1])}")
            else:
              accum_final_params += weight * math_int.exp(1j * final_params)
            bias_param = params[4]
            accum_bias += weight * bias_param
            total_weight_final += weight

    # 5. Compute the weighted circular averages.
    print_cust(f"federated_averaging_circular_parallel_shared, total_weight_conv: {total_weight_conv}, total_weight_final: {total_weight_final}")
    shared_conv_avg = []
    for i in range(min_conv_layers):
        avg_even = math_int.angle(shared_conv_even_accum[i] / total_weight_conv[i])
        avg_odd = math_int.angle(shared_conv_odd_accum[i] / total_weight_conv[i])
        shared_conv_avg.append((avg_even, avg_odd))
    shared_conv_avg = tuple(shared_conv_avg)

    shared_pool_avg = []
    for i in range(min_conv_layers):
        avg_pool = math_int.angle(shared_pool_accum[i] / total_weight_pool[i])
        shared_pool_avg.append(avg_pool)
    shared_pool_avg = tuple(shared_pool_avg)

    shared_final_pool_param = container_func([math_int.angle(accum_final_pool / total_weight_final)])
    if isinstance(any_client[3], (list, tuple)):
      print_cust(f"federated_averaging_circular_parallel_shared, type(accum_final_params[0]): {type(accum_final_params[0])}, type(accum_final_params[1]): {type(accum_final_params[1])}, type(total_weight_final): {type(total_weight_final)}")
      shared_final_params = (math_int.angle(accum_final_params[0] / total_weight_final), math_int.angle(accum_final_params[1] / total_weight_final))
    else:
      shared_final_params = math_int.angle(accum_final_params / total_weight_final)
    shared_bias = accum_bias / total_weight_final

    # NOTE: in the generative case (I do instanceof checks to check), this aggregates the generator params only.
    # forcing use of torch here. can change later

    if generative or min_conv_layers == 0:
      math_int = torch
      complex_dtype = torch.complex64
      float_dtype = torch.float32
      container_func = torch.tensor

    print_cust(f"federated_averaging_circular_parallel_shared, right before block params agg, math_int: {math_int}")

    with torch.no_grad():
      aggregated_block_params = aggregate_block_params_general(client_params_dict, clients_data_dict, math_int=math_int)
      shared_block_params = {}
      for block_type, (aggregated_common, aggregated_extra) in aggregated_block_params.items():
          shared_block_params[block_type] = aggregated_common
      if generative:
        # assumed, if generative, that (1) block_params contains discriminator, and (2) there exists at least one discriminator.
        # ALSO, this assumes that if a client is in the training, then its disc params need to be aggregated.
        agg_disc = aggregate_discriminator_params(client_params_dict, clients_data_dict)
        # this changes how the block params need to be used downstream dep on generative... not sure how else to adapt it otherwise for now
        shared_block_params = [shared_block_params, agg_disc]

    print_cust(f"federated_averaging_circular_parallel_shared, shared_block_params: {shared_block_params}")

    # 7. Return the aggregated shared parameters in the same 6-tuple structure.
    aggregated_shared_params = (
        shared_conv_avg,
        shared_pool_avg,
        shared_final_pool_param,
        shared_final_params,
        shared_bias,
        shared_block_params
    )
    return aggregated_shared_params