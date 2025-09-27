from quorus.logging.custom_slog import print_cust
import torch

"""## Single-Evaluation, Ancilla QCNN QNode Creation"""

def run_singleeval_qnode(input_angles, params, qnode):
  """
  Runs a multi-evaluation QNode of the specified size. Runs the QNode once, but with ancilla ubits.
  
  Parameters:
    input_angles: the data fed into the quantum model of interest
    params: a tuple of parameters to be called with this qnode.
    qnode: A Callable with inputs (input_angles, params)

  Returns:
    output_probs_tensors, a list of tensors containing the output predictions for the input.
  """
  print_cust(f"run_singleeval_qnode, params: {params}")
  output_probs_list = qnode(input_angles, params)
  # assumed to be a list of probabilities for each qubit output, because the QC should be using ancillas and return
  # a list of torch tensors.
  output_probs_tensors = torch.stack(output_probs_list)
  print_cust(f"run_singleeval_qnode, output_probs_tensors: {output_probs_tensors}, output_probs_tensors.shape: {output_probs_tensors.shape}")
  return output_probs_tensors