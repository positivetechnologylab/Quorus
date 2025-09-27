from pennylane import numpy as np
from quorus.logging.custom_slog import print_cust
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator
import torch

"""### Circular mean parallel"""

from collections import defaultdict

"""#### Block parameters aggregation function"""

def aggregate_block_params_general(client_params_dict, clients_data_dict, math_int=np):
    """
    Aggregates block (HEA ansatz) parameters for heterogeneous clients in a general way.

    Each client's block parameters (params[5]) is a list of numpy arrays,
    each with shape (num_layers, num_qubits, 3). We ignore the order in the list and
    group arrays by their block type (defined by bp.shape[1], i.e. the number of qubits).

    Args:
      client_params_dict: dict mapping client types (e.g., model sizes) to lists of parameter tuples.
          Each tuple is (conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list).
      clients_data_dict: dict mapping client types to lists of client data tuples.
          Each client data tuple is ((X_train, y_train), (X_val, y_val)).

    For each block type:
      - Let depths be the set of depths among all clients.
      - Define common_depth = min(depths). This is the number of bottom layers shared by all clients.
      - Define extra_count = max(depths) - common_depth.
      - For each client, its contribution to the common part is its bottom common_depth layers.
      - Its extra part is the top (depth - common_depth) layers.
        We align these extra layers in a meta extra region of length extra_count by computing an offset:
            offset = extra_count - (client_depth - common_depth)
        so that clients with fewer extra layers “line up” at the bottom.
      - Both the common and extra parts are aggregated using weighted circular averaging,
        with weight given by the client’s training set size.

    Returns:
      A dict mapping block type (number of qubits) to a tuple:
         (aggregated_common, aggregated_extra)
      where aggregated_common has shape (common_depth, num_qubits, 3) and
            aggregated_extra has shape (extra_count, num_qubits, 3) (or is None if no extra layers).
    """
    # Group block arrays by block type; record weight, depth, and array.
    print_cust(f"aggregate_block_params_general, math_int: {math_int}")
    block_groups = defaultdict(list)
    for ct, clients in client_params_dict.items():
        print_cust(f"aggregate_block_params_general, ct: {ct}")
        for params, data in zip(clients, clients_data_dict[ct]):
            weight = len(data[0][0])  # number of training samples for this client
            print_cust(f"aggregate_block_params_general, weight: {weight}")
            block_list = params[5]
            if block_list is None or len(block_list) == 0:
                continue
            if isinstance(block_list[0], PatchQuantumGenerator):
              block_list = block_list[0].q_params
            # NOTE: I MAY be breaking the block param functionality because I am saying that the ORDER of the block params matters.
            # Inducing additional structure; but helps for the case where you have SAME number of qubits, but
            # different LAYERS you want to optimize over for the BP's.
            for bp_idx, bp in enumerate(block_list):
                key = bp.shape[1]  # block type is determined by the number of qubits
                block_groups[(key, bp_idx)].append((weight, bp.shape[0], bp))

    aggregated_block = {}
    for key_tuple, group in block_groups.items():
        # For this block type (e.g. 4 qubits), determine the depths.
        key = key_tuple[0]
        bp_idx_group = key_tuple[1]
        depths = [item[1] for item in group]
        common_depth = min(depths)
        max_depth = max(depths)
        extra_count = max_depth - common_depth

        # Initialize accumulators for the common part:
        # Shape: (common_depth, key, 3)
        if math_int == np:
          common_accum = math_int.zeros((common_depth, key, 3), dtype=np.complex128)
        else:
          common_accum = math_int.zeros((common_depth, key, 3), dtype=torch.complex64)
        # We use a vector weight for each common layer;
        # adding a scalar to a vector broadcasts the scalar across all positions.

        if math_int == np:
          common_weight = math_int.zeros(common_depth, dtype=np.float64)
        else:
          common_weight = math_int.zeros(common_depth, dtype=torch.float32)

        # Initialize accumulators for the extra part (if any).
        if extra_count > 0:
            if math_int == np:
              extra_accum = math_int.zeros((extra_count, key, 3), dtype=np.complex128)
              extra_weight = math_int.zeros(extra_count, dtype=np.float64)
            else:
              extra_accum = math_int.zeros((extra_count, key, 3), dtype=torch.complex64)
              extra_weight = math_int.zeros(extra_count, dtype=torch.float32)
        else:
            extra_accum = None
            extra_weight = None

        # Loop over contributions.
        for weight, depth, bp in group:
            # Common part: use the bottom common_depth layers from this client's array.
            print_cust(f"aggregate_block_params_general, weight (in group loop): {weight}")
            common_part = bp[depth - common_depth : depth, :, :]  # shape (common_depth, key, 3)
            common_accum += weight * math_int.exp(1j * common_part)
            common_weight += weight  # scalar addition broadcasts to each common layer

            # Extra part: if this client has extra layers.
            client_extra = depth - common_depth
            if client_extra > 0 and extra_count > 0:
                # Align extra part: offset = extra_count - (client_extra)
                offset = extra_count - client_extra
                extra_part = bp[0 : client_extra, :, :]  # shape (client_extra, key, 3)
                # NOTE, layers: this is weird. Offset gets larger as an idx, BUT I'm (prev line) getting FIRST client_extra layers of clients' block. can change this later I guess.
                extra_accum[offset : offset + client_extra] += weight * math_int.exp(1j * extra_part)
                extra_weight[offset : offset + client_extra] += weight

        print_cust(f"aggregate_block_params_general, common_weight: {common_weight}")
        aggregated_common = math_int.angle(common_accum / common_weight[:, None, None])
        if extra_count > 0:
            aggregated_extra = math_int.angle(extra_accum / extra_weight[:, None, None])
        else:
            aggregated_extra = None
        aggregated_block[key_tuple] = (aggregated_common, aggregated_extra)

    print_cust(f"aggregate_block_params_general, aggregated_block: {aggregated_block}")

    for test_key_tuple, (test_common, test_extra) in aggregated_block.items():
      print_cust(f"aggregate_block_params_general, test_key_tuple: {test_key_tuple}")
      test_key = test_key_tuple[0]
      test_key_bpidx = test_key_tuple[1]
      print_cust(f"aggregate_block_params_general, test_key: {test_key}")
      if test_common is not None:
        print_cust(f"aggregate_block_params_general, test_common: {test_common}")
        print_cust(f"aggregate_block_params_general, test_common.requires_grad: {test_common.requires_grad}")
        print_cust(f"aggregate_block_params_general, test_common.grad_fn: {test_common.grad_fn}")
        if test_common.grad_fn is not None:
          print_cust(f"aggregate_block_params_general, test_common.grad_fn.next_functions: {test_common.grad_fn.next_functions}")
      if test_extra is not None:
        print_cust(f"aggregate_block_params_general, test_extra: {test_extra}")
        print_cust(f"aggregate_block_params_general, test_extra.requires_grad: {test_extra.requires_grad}")
        print_cust(f"aggregate_block_params_general, test_extra.grad_fn: {test_extra.grad_fn}")
        if test_extra.grad_fn is not None:
          print_cust(f"aggregate_block_params_general, test_extra.grad_fn.next_functions: {test_extra.grad_fn.next_functions}")

    return aggregated_block