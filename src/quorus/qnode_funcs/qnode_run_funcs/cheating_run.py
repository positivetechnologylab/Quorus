from quorus.logging.custom_slog import print_cust
import pennylane as qml
import torch

"""## Single-Evaluation, Cheating QCNN QNode Creation"""

def run_multiprob_qnode_cheating(input_angles, params, qnode):
  """
  Runs a multi-evaluation QNode of the specified size. Runs the QNode once, with cheating measurements (measures a qubit without collapsing the
  quantum state)

  Parameters:
    input_angles: the data fed into the quantum model of interest
    params: a tuple of parameters to be called with this qnode.
    qnode: A Callable with inputs (input_angles, params)

  Returns:
    output_probs_tensors, a list of tensors containing the output predictions for the input.
  """
  print_cust(f"run_multiprob_qnode_cheating, params: {params}")
  for block_param_idx, block_param in enumerate(params[5]):
    print_cust(f"run_multiprob_qnode_cheating, block_param_idx: {block_param_idx}, block_param: {block_param}")
  snapshot_results = qml.snapshots(qnode)(input_angles, params)
  list_results = []
  for block_param_idx in range(len(params[5])):
    list_results.append(snapshot_results[f"p0_bp{block_param_idx}"])
  # assumed to be a list of probabilities for each qubit output, because the QC should be using ancillas and return
  # a list of torch tensors.
  # TODO: postprocess differently, to get the cheating statistics.
  output_probs_tensors = torch.stack(list_results)
  print_cust(f"run_multiprob_qnode_cheating, output_probs_tensors: {output_probs_tensors}, output_probs_tensors.shape: {output_probs_tensors.shape}")
  return output_probs_tensors