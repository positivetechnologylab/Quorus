"""### Helper for unpermuting inputs"""

def unpermute_inputs(inputs, indices):
  """
  Unpermutes a list of numbers given indices.

  Parameters:
    inputs: a list of values
    indices: a list of integers representing indices of inputs

  Returns:
    a list of inputs that are the unpermuted inputs.
  """
  new_inputs = [0 for _ in range(len(inputs))]
  for orig_input_idx, targ_input_idx in enumerate(indices):
    new_inputs[targ_input_idx] = inputs[orig_input_idx]
  return new_inputs