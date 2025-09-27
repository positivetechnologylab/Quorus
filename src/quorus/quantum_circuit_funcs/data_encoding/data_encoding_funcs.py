from quorus.logging.custom_slog import print_cust
import pennylane as qml

"""# Angle encoding circuit"""
def pennylane_angleencode(qubits, inputs, reps=1, generative=False, axis="Y", is_reuploading=False):
    """
    Perform angle encoding or amplitude encoding based on the size of the inputs
    with respect to the number of qubits.

    Parameters
    ----------
    qubits : Sequence[QubitIdentifier]
        The qubits/wires to encode the data onto.
    inputs : 1D or 2D array
        The inputs to encode.
    reps : int
        The number of times to encode the data (for angle encoding specifically).

    Returns
    -------
    None (modfifies qml to add operations to a quantum circuit)
    """
    print_cust(f"pennylane_angleencode, axis: {axis}")
    print_cust(f"pennylane_angleencode, generative: {generative}")
    if len(inputs.shape) == 1:
      input_dim = len(inputs)
    elif len(inputs.shape) == 2:
      input_dim = inputs.shape[1]
    print_cust(f"pennylane_angleencode, input_dim: {input_dim}")
    # NOTE, layers: changed the logic here. only if can fit the data on qubits, do I do angle encoding.
    if input_dim <= len(qubits):
      print_cust(f"pennylane_angleencode, doing angle encoding")
      for rep in range(reps):
        if len(inputs.shape) == 1:
          for qubit_idx in range(len(qubits)):
            qubit = qubits[qubit_idx]
            if generative:
              qml.RX(inputs[qubit_idx], wires=qubit)
            qml.RY(inputs[qubit_idx], wires=qubit)
        elif len(inputs.shape) == 2:
          print_cust(f"pennylane_angleencode, running qml.templates.AngleEmbedding")
          if generative:
            qml.templates.AngleEmbedding(inputs, wires=qubits, rotation="X")
          qml.templates.AngleEmbedding(inputs, wires=qubits, rotation=axis)
    else:
      if not is_reuploading:
        print_cust(f"pennylane_angleencode, doing amplitude encoding")
        qml.AmplitudeEmbedding(
            features=inputs,
            wires=qubits,
            pad_with=0,
            normalize=True
        )