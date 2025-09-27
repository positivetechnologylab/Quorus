from pennylane import numpy as np
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator
from quorus.logging.custom_slog import print_cust
from quorus.qfl_utils.aggregation_funcs.agg_utils.block_params_agg import aggregate_block_params_general

"""## Meta-params initialization"""

def generate_meta_params(client_params_dict, clients_data_dict, math_int=np):
    """
    Generate meta parameters that cover all shared parameters across clients.

    For the convolutional, pooling, final pooling, final rotation, and bias parameters, we use
    the parameters from the client with the largest model type (which is a superset for these layers).
    For the block parameters, we aggregate them from all clients using the aggregate_block_params_general function.

    Parameters:
      client_params_dict: dict mapping client types (e.g., model sizes) to lists of parameter tuples.
          Each tuple is (conv_params, pool_params, final_pool_param, final_params, bias_param, block_params_list).
      clients_data_dict: dict mapping client types to lists of client data tuples.
          Each client data tuple is ((X_train, y_train), (X_val, y_val)).
      n_output_qubits: an integer representing the number of output qubits for the QCNN model.

    Returns:
        A tuple (meta_conv, meta_pool, meta_final_pool, meta_final, meta_bias, meta_block)
        that represents the meta parameter set.
    """
    # Identify the largest model type (assumes keys are numerical sizes).
    print_cust(f"generate_meta_params, math_int: {math_int}")
    max_model_type = max(client_params_dict.keys())
    # Use the parameters from the first client of the largest model type.
    base_meta = client_params_dict[max_model_type][0]
    meta_conv, meta_pool, meta_final_pool, meta_final, meta_bias, meta_block_orig = base_meta

    meta_block = aggregate_block_params_general(client_params_dict, clients_data_dict, math_int=math_int)
    print_cust(f"generate_meta_params, meta_block: {meta_block}")
    for block_type, (aggregated_common, aggregated_extra) in meta_block.items():
      # NOTE: now, block_type is a tuple.
      print_cust(f"generate_meta_params, type(block_type): {type(block_type)}, type(aggregated_common): {type(aggregated_common)}, type(aggregated_extra): {type(aggregated_extra)}")

    # Aggregate block parameters from all clients.
    if isinstance(meta_block_orig[0], PatchQuantumGenerator):
      print_cust(f"generate_meta_params, meta_block_orig[0] is a PatchQuantumGenerator")
      # Placeholder for now; assumes homogenous architecture among all clients.
      # meta_gen_params = aggregate_block_params_general(client_params_dict, clients_data_dict, math_int=math_int)
      meta_disc = meta_block_orig[1]
      meta_block = [meta_block, meta_disc]
      print_cust(f"generate_meta_params, generator case, meta_block: {meta_block}")

    # Construct and return the meta parameter tuple.
    return (meta_conv, meta_pool, meta_final_pool, meta_final, meta_bias, meta_block)