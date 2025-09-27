from pennylane import numpy as np
import torch
from quorus.logging.custom_slog import print_cust

"""## Random meta params initialization"""

# TODO: continue here; generate_meta_params_random. See how it is used in the overall training code.
def generate_meta_params_random(meta_params, math_int=np):
    """
    Given meta_params with structure:
      (meta_conv, meta_pool, meta_final_pool, meta_final, meta_bias, meta_block)
    where:
      - meta_conv is a tuple of layers, each a tuple (even_params, odd_params)
      - meta_pool is a tuple of pooling parameter arrays
      - meta_final_pool is a numpy array (e.g., shape (1,))
      - meta_final is a numpy array (e.g., shape (1,3))
      - meta_bias is a numpy array
      - meta_block is a dict mapping block type (int) to a tuple
          (aggregated_common, aggregated_extra), where aggregated_common is an array
          of shape (common_depth, num_qubits, 3) and aggregated_extra is either None
          or an array of shape (extra_layers, num_qubits, 3)

    Returns:
      meta_params_random: a new tuple with the same structure as meta_params,
                          but with each numpy array re-initialized with random values.
    """
    meta_conv, meta_pool, meta_final_pool, meta_final, meta_bias, meta_block = meta_params

    rand_int = np.random

    # Generate random meta_conv with the same structure.
    random_meta_conv = tuple(
        (rand_int.randn(*even.shape), rand_int.randn(*odd.shape))
        for even, odd in meta_conv
    )

    # Generate random meta_pool with the same structure.
    random_meta_pool = tuple(
        rand_int.randn(*pool.shape) for pool in meta_pool
    )

    # Generate random arrays for final pooling, final rotation, and bias.
    random_meta_final_pool = rand_int.randn(*meta_final_pool.shape)
    if isinstance(meta_final, (list, tuple)):
      random_meta_final = (rand_int.randn(*meta_final[0].shape), rand_int.randn(*meta_final[1].shape))
    else:
      random_meta_final = rand_int.randn(*meta_final.shape)
    random_meta_bias = rand_int.randn(*meta_bias.shape)

    # For block parameters, generate a random array for each block type.
    # meta_block is assumed to be a dict mapping block type to (aggregated_common, aggregated_extra)

    meta_block_dict = meta_block

    if math_int == np:
      rand_int = np.random
    else:
      rand_int = torch

    if not isinstance(meta_block, dict):
      meta_block_dict = meta_block[0]
      assert isinstance(meta_block_dict, dict), f"generate_meta_params_random, meta_block_dict is not a dict, type(meta_block_dict): {type(meta_block_dict)}"

    random_meta_block = {}
    for block_type, (common, extra) in meta_block_dict.items():
        random_common = rand_int.randn(*common.shape)
        random_extra = rand_int.randn(*extra.shape) if extra is not None else None
        random_meta_block[block_type] = (random_common, random_extra)

    if not isinstance(meta_block, dict):
      print_cust(f"generate_meta_params_random, meta_block is not a dict")
      random_meta_block = [random_meta_block, meta_block[1]]

    print_cust(f"generate_meta_params_random, random_meta_block: {random_meta_block}")


    return (random_meta_conv, random_meta_pool, random_meta_final_pool,
            random_meta_final, random_meta_bias, random_meta_block)