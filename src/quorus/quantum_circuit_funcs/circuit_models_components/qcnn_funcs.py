import pennylane as qml

from pennylane import numpy as np

def conv_layer(params, wires, encoded_angles=None, reversed_indices=None, pool_in=False):
    """
    A convolutional layer applying rotations and CNOTs on pairs of qubits.

    Parameters:
      params: a tuple (even_params, odd_params) where:
         - even_params is a numpy array of shape (num_even_pairs, 12) applied to pairs:
           (wires[0], wires[1]), (wires[2], wires[3]), ...
         - odd_params is a numpy array of shape (num_odd_pairs, 12) applied to pairs:
           (wires[1], wires[2]), (wires[3], wires[4]), ...
      wires: list of qubit indices.
      encoded_angles: list of # TODO
      reversed_indices: list of qubit indices for which to apply a reverse operation (used in identity initialization)
      pool_in: boolean indicating whether or not the qubits should pool inward (not used)

    For each block:
      1. Apply a first set of rotations (using parameters indices 0–5) on the two qubits.
      2. Apply a CNOT gate (entangling).
      3. Apply a second set of rotations (using parameters indices 6–11) on the two qubits.
      4. Apply a second CNOT (with reversed control).

    Returns:
      None (modifies qml to add operations)
    """
    num_wires = len(wires)

    # Process even pairs: indices 0,1; 2,3; etc.
    num_even_pairs = num_wires // 2
    for idx in range(num_even_pairs):
        i = idx * 2
        # First block of rotations on the even pair.
        qml.Rot(params[0][idx, 0], params[0][idx, 1], params[0][idx, 2], wires=wires[i])
        qml.Rot(params[0][idx, 3], params[0][idx, 4], params[0][idx, 5], wires=wires[i + 1])

        if encoded_angles is not None:
          # reversed indices is the QUBIT indices.
          if i in reversed_indices:
            qml.RY(-1 * encoded_angles[i], wires=wires[i])
          if (i + 1) in reversed_indices:
            qml.RY(-1 * encoded_angles[i + 1], wires=wires[i + 1])

        qml.CNOT(wires=[wires[i], wires[i + 1]])
        # Second block of rotations.
        qml.Rot(params[0][idx, 6], params[0][idx, 7], params[0][idx, 8], wires=wires[i])
        qml.Rot(params[0][idx, 9], params[0][idx, 10], params[0][idx, 11], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i + 1], wires[i]])

    # Process odd pairs: pairs (wires[1], wires[2]), (wires[3], wires[4]), etc.
    num_odd_pairs = (num_wires - 1) // 2
    for idx in range(num_odd_pairs):
        i = 1 + idx * 2
        # First block of rotations for odd pair.
        qml.Rot(params[1][idx, 0], params[1][idx, 1], params[1][idx, 2], wires=wires[i])
        qml.Rot(params[1][idx, 3], params[1][idx, 4], params[1][idx, 5], wires=wires[i + 1])
        if encoded_angles is not None:
          if (i + 1) == (num_wires - 1):
            if (i + 1) in reversed_indices:
              qml.RY(-1 * encoded_angles[i + 1], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i], wires[i + 1]])
        # Second block of rotations.
        qml.Rot(params[1][idx, 6], params[1][idx, 7], params[1][idx, 8], wires=wires[i])
        qml.Rot(params[1][idx, 9], params[1][idx, 10], params[1][idx, 11], wires=wires[i + 1])
        qml.CNOT(wires=[wires[i + 1], wires[i]])

    # If there is an extra wire at the end, apply an identity rotation.
    if num_wires % 2 == 1:
        qml.Rot(0.0, 0.0, 0.0, wires=wires[-1])

def pool_layer_with_measurement(params, wires, pool_in=False):
    """
    Parameters:
      params: a np.ndarray of parameters
      wires: list of qubit indices.
      pool_in: boolean indicating whether or not the qubits should pool inward (not used)

    Pooling layer using mid-circuit measurements.
    For each consecutive pair, measures the second qubit and conditionally applies RY on the first.
    Expects params of shape (number_of_pairs, 1) where number_of_pairs = floor(len(wires)/2).
    Returns a list of the retained (control) wires.
    """
    pooled_wires = []
    num_pairs = len(wires) // 2
    for i in range(0, num_pairs * 2, 2):
      if pool_in:
        if i < (num_pairs):
          m = qml.measure(wires[i])
          qml.cond(m, lambda w=wires[i + 1]: qml.RY(np.pi, wires=w))
          # qml.cond(m, lambda w=wires[i]: qml.RY(params[i // 2, 0], wires=w))
          pooled_wires.append(wires[i + 1])
        else:
          m = qml.measure(wires[i + 1])
          qml.cond(m, lambda w=wires[i]: qml.RY(np.pi, wires=w))
          pooled_wires.append(wires[i])
      else:
        m = qml.measure(wires[i + 1])
        qml.cond(m, lambda w=wires[i]: qml.RY(np.pi, wires=w))
        # qml.cond(m, lambda w=wires[i]: qml.RY(params[i // 2, 0], wires=w))
        pooled_wires.append(wires[i])
    if len(wires) % 2 == 1:
        pooled_wires.append(wires[-1])
    return pooled_wires

def final_pool(wires, param):
    """
    Parameters:
      wires: list of qubit indices.
      param: a float indicating the parameter to apply for pooling.

    Final pooling: measures the second qubit and conditionally rotates the first.
    Returns the first wire.
    """
    m = qml.measure(wires[1])
    qml.cond(m, lambda w=wires[0]: qml.RY(np.pi, wires=w))
    # qml.cond(m, lambda w=wires[0]: qml.RY(param, wires=w))
    return wires[0]