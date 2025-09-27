from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
from quorus.qfl_utils.aggregation_funcs.agg_utils.convert_block_params_generator import convert_agg_block_params_generator
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator
import torch

"""### Broadcast parameters function, all aggregate"""

def broadcast_param_updates(client_params_dict, aggregated_params, math_int=np):
    """

    Args:
      client_params_dict: dict mapping client types (e.g. model sizes) to lists of parameter tuples.
    aggregated_params is assumed to have the structure:
      (meta_avg_conv, meta_avg_pool, avg_final_pool_param, avg_final_params, avg_bias, aggregated_block_params)
      and contains the aggregated parameters to broadcast to the clients.

    Update the clients' parameters in client_params_dict using the aggregated_params.



      - meta_avg_conv: tuple of length meta_conv_layers; each element is a tuple (even, odd)
      - meta_avg_pool: tuple of length meta_conv_layers; each element is an array
      - avg_final_pool_param: numpy array (1,)
      - avg_final_params: numpy array of shape (1,3) or similar
      - avg_bias: numpy array (bias scalar or vector)
      - aggregated_block_params: a dict mapping block type (num_qubits) to a tuple:
            (aggregated_common, aggregated_extra)
          where aggregated_common is a numpy array of shape (common_depth, num_qubits, 3) and
          aggregated_extra is either None or a numpy array of shape (meta_extra, num_qubits, 3).

    For the convolution and pooling parameters, each client’s parameters are updated based on its
    “offset” into the meta (global) parameter set.

    For the block parameters, for each block array bp (shape (client_depth, num_qubits, 3)) in the client's
    block_params_list:
      - The bottom common_depth layers are replaced by the aggregated common block for that block type.
      - If bp has extra layers (client_depth > common_depth) and an aggregated extra block exists,
        then bp’s top extra layers are updated. They are aligned using:
              offset = meta_extra - (client_depth - common_depth)
        so that the bottom of the extra region is shared among all clients.

    The function updates client_params_dict in place and returns it.
    """

    meta_avg_conv, meta_avg_pool, avg_final_pool_param, avg_final_params, avg_bias, aggregated_block_params = aggregated_params
    meta_conv_layers = len(meta_avg_conv)
    if not isinstance(aggregated_block_params, dict):
      aggregated_block_params = convert_agg_block_params_generator(aggregated_block_params)
      print_cust(f"broadcast_param_updates, aggregated_block_params: {aggregated_block_params}")

    # TODO, 7/11: load in the generator, disc params into the model for gen, disc models.

    # Iterate over each client type and update its clients.
    for client_type in client_params_dict:
        for i, params in enumerate(client_params_dict[client_type]):
            conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list = params

            # --- Update convolution parameters ---
            client_conv_layers = len(conv_params)
            offset_conv = meta_conv_layers - client_conv_layers
            new_conv_params = []
            for l in range(client_conv_layers):
                # Replace client's conv layer l with meta conv layer at l+offset_conv.
                new_conv_params.append(meta_avg_conv[l + offset_conv])

            # --- Update pooling parameters ---
            client_pool_layers = len(pool_params)
            offset_pool = meta_conv_layers - client_pool_layers
            new_pool_params = []
            for l in range(client_pool_layers):
                new_pool_params.append(meta_avg_pool[l + offset_pool])

            # --- Update final pooling, final rotation, and bias ---
            new_final_pool_param = np.array(avg_final_pool_param)  # shallow copy
            if isinstance(avg_final_params, tuple):
              new_final_params = tuple(avg_final_params)
            else:
              new_final_params = np.array(avg_final_params)
            new_bias_param = np.array(avg_bias)

            # --- Update block (HEA ansatz) parameters ---
            new_block_params_list = []
            # For each block parameter array bp in the list:
            if not isinstance(block_params_list[0], PatchQuantumGenerator):
              with torch.no_grad():
                for bp_idx, bp in enumerate(block_params_list):
                    # bp has shape (client_depth, num_qubits, 3)
                    client_depth, num_qubits, _ = bp.shape
                    # If this block type (num_qubits) has aggregated updates:
                    if (num_qubits, bp_idx) in aggregated_block_params:
                        aggregated_common, aggregated_extra = aggregated_block_params[(num_qubits, bp_idx)]
                        common_depth = aggregated_common.shape[0]  # minimum depth shared among all clients for this block type
                        if math_int == np:
                          new_bp = bp.copy()
                        elif math_int == torch:
                          print_cust(f"broadcast_param_updates, applying .detach().clone() to bp")
                          new_bp = bp.detach().clone()
                        # Update common part: last common_depth layers of bp.
                        # NOTE: in general, I do NOT want to update the last common layers; I want to update the FIRST
                        # commmon layers. but, b/c ACROSS clients of the same type, the layers should always be the same
                        # structure, it does not matter.
                        new_bp[client_depth - common_depth : client_depth, :, :] = aggregated_common
                        # Update extra part if bp has extra layers and aggregated_extra is available.
                        client_extra = client_depth - common_depth
                        if client_extra > 0 and (aggregated_extra is not None):
                            meta_extra = aggregated_extra.shape[0]
                            # Align: offset = meta_extra - client_extra
                            offset_extra = meta_extra - client_extra
                            new_bp[0:client_extra, :, :] = aggregated_extra[offset_extra : offset_extra + client_extra]
                        new_block_params_list.append(new_bp)
                    else:
                        # No aggregated update for this block type; leave unchanged.
                        print_cust(f"broadcast_param_updates, no update for bp bp_idx: {bp_idx}, client_depth: {client_depth}, num_qubits: {num_qubits}")
                        new_block_params_list.append(bp)
            else:
              # This is the generator case
              existing_generator = block_params_list[0]
              existing_discriminator = block_params_list[1]
              with torch.no_grad():
                existing_generator.load_state_dict(aggregated_block_params[0].state_dict())
                print_cust(f"broadcast_param_updates, existing_generator: {existing_generator}")
                # print_cust(f"broadcast_param_updates, existing_generator.state_dict(): {existing_generator.state_dict()}")
                existing_discriminator.load_state_dict(aggregated_block_params[1].state_dict())
              print_cust(f"broadcast_param_updates, existing_discriminator: {existing_discriminator}")
              # print_cust(f"broadcast_param_updates, existing_discriminator.state_dict(): {existing_discriminator.state_dict()}")
              new_block_params_list = block_params_list

            # Update the client parameter tuple.
            client_params_dict[client_type][i] = (tuple(new_conv_params),
                                                    tuple(new_pool_params),
                                                    new_final_pool_param,
                                                    new_final_params,
                                                    new_bias_param,
                                                    new_block_params_list)
    return client_params_dict