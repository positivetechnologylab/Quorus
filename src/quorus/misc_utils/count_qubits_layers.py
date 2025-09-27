def count_qubits_layers(qubits_layers_list):
  """
  Function that counts the layers for each qubit/client type.

  Parameters:
      qubits_layers_list (List[Tuple(int, int)]): A list of tuples (num_qubits, num_layers) specifying, for all clients of a particular client type,
      the number of layers that they have.

  Returns:
      A string representing the qubits and layer types for this client.
  """
  list_counts = []
  for qubits_and_layers in qubits_layers_list:
    list_counts.append(str(qubits_and_layers[0]))
    list_counts.append(str(qubits_and_layers[1]))
  return "_".join(list_counts)