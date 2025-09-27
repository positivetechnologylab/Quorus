from quorus.logging.custom_slog import print_cust

"""# Conv layers computation helper funcs

## Conv layers computation function
"""

def compute_conv_layers(n_qubits, n_output_qubits, generative=False, is_qcnn=True):
    """
    Given the number of input qubits and the number of output qubits for a QCNN, computes the
    number of convolutional layers in the QCNN. Assumes that, after each convolutional layer, the number
    of qubits in the circuit is halved.

    Parameters:
      n_qubits (int): The number of input qubits to the QCNN.
      n_output_qubits (int): The number of output qubits in the QCNN (these are likely measured to obtain your final result).

    Returns:
      conv_layers (int): The number of convolutional layers in your QCNN.
    """
    if generative or not is_qcnn:
      print_cust(f"compute_conv_layers, returning 0")
      return 0
    conv_layers = 0
    # Start with all your qubits.
    temp = n_qubits
    # While you haven't reached your output, halve the number of qubits.
    while temp >= n_output_qubits:
        print_cust(f"compute_conv_layers, temp: {temp}, n_output_qubits: {n_output_qubits}")
        if n_output_qubits == 1 and temp == 1:
          break
        # If pooling with an odd number, keep the extra qubit.
        if temp % 2 == 0:
            temp = temp // 2
        else:
            temp = temp // 2 + 1
        if temp >= n_output_qubits:
          conv_layers += 1
    return conv_layers

# TOADD: an argument for the function above, for generative QFL, to just return 0.
# DONE: TOMODIFY, layers: an additional argument that specifies if we are operating in the non-conv layers case, in which case, we return 0.

"""## Num qubits computation function"""

def compute_reduced_qubits(n_qubits, conv_layers):
  cur_num_qubits = n_qubits
  for _ in range(conv_layers):
    if cur_num_qubits % 2 == 0:
      cur_num_qubits = cur_num_qubits // 2
    else:
      cur_num_qubits = cur_num_qubits // 2 + 1
  return cur_num_qubits