import copy
from quorus.logging.custom_slog import print_cust
import torch

"""## Multi-Evaluation QCNN QNode Creation"""

def run_multiprob_qnode(input_angles, params, qnode):
  """
  Runs a multi-evaluation QNode of the specified size. Runs the QNode multiple times.
  
  Parameters:
    input_angles: the data fed into the quantum model of interest
    params: a tuple of parameters to be called with this qnode.
    qnode: A Callable with inputs (input_angles, params)

  Returns:
    output_probs_tensors, a list of tensors containing the output predictions for the input.
  """
  output_probs_list = []
  block_params_list = params[5]
  for block_param_idx in range(1, len(block_params_list) + 1):
    params_copy = list(copy.deepcopy(params))
    sub_blockparams_list = block_params_list[:block_param_idx]
    params_copy[5] = sub_blockparams_list
    params_copy = tuple(params_copy)
    print_cust(f"run_multiprob_qnode, block_param_idx: {block_param_idx}, params_copy: {params_copy}")
    output_probs = qnode(input_angles, params_copy)
    output_probs_list.append(output_probs)
  output_probs_tensors = torch.stack(output_probs_list)
  print_cust(f"run_multiprob_qnode, output_probs_tensors: {output_probs_tensors}, output_probs_tensors.shape: {output_probs_tensors.shape}")
  return output_probs_tensors