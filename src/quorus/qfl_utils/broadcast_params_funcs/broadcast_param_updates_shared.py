from quorus.qfl_utils.aggregation_funcs.agg_utils.convert_block_params_generator import convert_agg_block_params_generator
from pennylane import numpy as np
from quorus.logging.custom_slog import print_cust
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator
import torch

"""### Broadcast parameters function, aggregate shared only"""

def broadcast_param_updates_shared(client_params_dict, aggregated_params, math_int=np):
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

    # Unpack aggregated shared parameters.
    shared_conv_avg, shared_pool_avg, shared_final_pool_param, shared_final_params, shared_bias, shared_block_params = aggregated_params

    if not isinstance(shared_block_params, dict):
      shared_block_params = convert_agg_block_params_generator(shared_block_params)
      print_cust(f"broadcast_param_updates_shared, shared_block_params: {shared_block_params}")
      # print_cust(f"broadcast_param_updates, shared_block_params[0].state_dict(): {shared_block_params[0].state_dict()}")
      # print_cust(f"broadcast_param_updates, shared_block_params[1].state_dict(): {shared_block_params[1].state_dict()}")

    # Number of shared convolution and pooling layers (these are the "common" layers to update).
    num_shared_conv = len(shared_conv_avg)
    num_shared_pool = len(shared_pool_avg)

    print_cust(f"broadcast_param_updates_shared, num_shared_conv: {num_shared_conv}, num_shared_pool: {num_shared_pool}")

    # Loop over each client type and each client.
    for client_type in client_params_dict:
        print_cust(f"broadcast_param_updates_shared, client_type: {client_type}")
        for i, params in enumerate(client_params_dict[client_type]):
            print_cust(f"broadcast_param_updates_shared, i: {i}")
            conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list = params

            # --- Update QCNN convolution parameters ---
            # conv_params is a tuple; update only the last 'num_shared_conv' layers.
            client_conv_layers = len(conv_params)
            offset_conv = client_conv_layers - num_shared_conv
            print_cust(f"broadcast_param_updates_shared, offset_conv: {offset_conv}")
            new_conv_params = list(conv_params)  # convert to list for modification
            for j in range(num_shared_conv):
                # Each conv layer is a tuple (even, odd) and we simply replace it.
                new_conv_params[offset_conv + j] = shared_conv_avg[j]

            # --- Update pooling parameters ---
            client_pool_layers = len(pool_params)
            offset_pool = client_pool_layers - num_shared_pool
            print_cust(f"broadcast_param_updates_shared, offset_pool: {offset_pool}")
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
            if not isinstance(block_params_list[0], PatchQuantumGenerator):
              with torch.no_grad():
                for bp_idx, bp in enumerate(block_params_list):
                    # Determine the block type by number of qubits (assumed bp.shape = (depth, n_qubits, 3)).
                    block_type = bp.shape[1]
                    if (block_type, bp_idx) in shared_block_params:
                        aggregated_common = shared_block_params[(block_type, bp_idx)]  # shape: (common_depth, block_type, 3)
                        common_depth = aggregated_common.shape[0]
                        print_cust(f"broadcast_param_updates_shared, common_depth: {common_depth}")
                        client_depth = bp.shape[0]
                        if math_int == np:
                          bp_updated = bp.copy()
                        elif math_int == torch:
                          print_cust(f"broadcast_param_updates_shared, applying .detach().clone() to bp")
                          bp_updated = bp.detach().clone()
                        # Update the last 'common_depth' layers.
                        bp_updated[client_depth - common_depth : client_depth, :, :] = aggregated_common
                        new_block_params_list.append(bp_updated)
                    else:
                        # If no update for this block type is provided, leave it unchanged.
                        print_cust(f"broadcast_param_updates_shared, no update for bp_idx: {bp_idx}, client_type: {client_type}, i: {i}")
                        new_block_params_list.append(bp)
            else:
              # This is the generator case
              existing_generator = block_params_list[0]
              existing_discriminator = block_params_list[1]
              with torch.no_grad():
                existing_generator.load_state_dict(shared_block_params[0].state_dict())
                print_cust(f"broadcast_param_updates_shared, existing_generator: {existing_generator}")
                # print_cust(f"broadcast_param_updates, existing_generator.state_dict(): {existing_generator.state_dict()}")
                existing_discriminator.load_state_dict(shared_block_params[1].state_dict())
              print_cust(f"broadcast_param_updates_shared, existing_discriminator: {existing_discriminator}")
              # print_cust(f"broadcast_param_updates, existing_discriminator.state_dict(): {existing_discriminator.state_dict()}")
              new_block_params_list = block_params_list

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