from quorus.logging.custom_slog import print_cust
import pennylane as qml

"""## Block circuit"""
def block_variational_circuit(params, layers, wires, generative=False, id_init_circ=False, is_hea=False, circuit_type="staircase",
                              reupload_func=None, inv_reupload_func=None, offset_idx=None):
    """
    Parameters:
      params: Array of shape (layers, len(wires), 3)
      layers: Number of layers in the block (should equal params.shape[0])
      wires: List of qubit indices.
      generative: a Boolean indicating whether or not this is a block in a QGAN.
      id_init_circ: a boolean indicating whether or not identity initialization should be applied in this block.
      is_hea: a Boolean indicating whether or not only nearest neighbor connectiviy is allowed.
      circuit_type: a string representing the type of the circuit used in the block.
      reupload_func: a function that reuploads the data onto the quantum circuit.
      inv_reupload_func: a function that reuploads the inverse of the data onto the quantum circuit.
      offset_idx: an integer representing the offset integer for this block.

    Applies a staircase hardware-efficient variational block.
    For each layer in the block:
      - Applies a qml.Rot gate (with 3 parameters) on each qubit in 'wires'.
      - Applies a CNOT on qubits, with structure specified by circuit_type.

    Returns:
      None (modifies qml by adding circuit operations)
    """
    print_cust(f"block_variational_circuit, params.shape: {params.shape}")
    print_cust(f"block_variational_circuit, layers: {layers}")
    print_cust(f"block_variational_circuit, circuit_type: {circuit_type}")
    print_cust(f"block_variational_circuit, reupload_func: {reupload_func}")
    print_cust(f"block_variational_circuit, inv_reupload_func: {inv_reupload_func}")
    print_cust(f"block_variational_circuit, offset_idx: {offset_idx}")
    print_cust(f"block_variational_circuit, wires: {wires}")
    for layer in range(layers):
        print_cust(f"block_variational_circuit, layer: {layer}")
        if reupload_func is not None:
          print_cust(f"block_variational_circuit, applying reupload_func()")
          reupload_func()
        # Apply single-qubit rotations.
        # NOTE, layers: in general, block params is NOT assumed to have the same number of qubits as the
        # number of wires.
        for i in range(params.shape[1]):
            wire = wires[i]
            qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=wire)
        # Apply the staircase of CNOTs.
        # NOTE, layers: in general, block params is NOT assumed to have the same number of qubits as the
        # number of wires.
        num_qubits = params.shape[1]
        n = num_qubits
        # if generative or is_hea:
        #   n = n - 1
        # TOMODIFY, layers: get rid of this below hack....
        # TOMODIFY, layers, HACK: generally this is NOT what I want, BUT I have it here to allow for final tuning in ID init.
        if ((layer + 1) % 3 == 0) and (layers % 3 == 0):
          print_cust(f"block_variational_circuit, layer: {layer}, layers: {layers}, skipping CNOT applications")
          continue
        print_cust(f'block_variational_circuit, n: {n}, num_qubits: {num_qubits}')
        if circuit_type == "staircase":
          if not id_init_circ or (layer % 2 == 0):
            print_cust(f"block_variational_circuit, staircase, not id_init_circ, even layer, layer: {layer}")
            for i in range(n):
                control = wires[i]
                target = wires[(i+1) % num_qubits]
                if i == (n - 1):
                  if not generative and not is_hea:
                    qml.CNOT(wires=[control, target])
                else:
                  qml.CNOT(wires=[control, target])
          elif id_init_circ and (layer % 2 == 1):
            print_cust(f"block_variational_circuit, staircase, id_init_circ, odd layer, layer: {layer}")
            for i in range(n - 1, -1, -1):
              control = wires[i]
              target = wires[(i+1) % num_qubits]
              if i == (n - 1):
                if not generative and not is_hea:
                  qml.CNOT(wires=[control, target])
              else:
                qml.CNOT(wires=[control, target])
        elif circuit_type == "brick":
          if not id_init_circ or (layer % 2 == 0):
            print_cust(f"block_variational_circuit, brick, not id_init_circ, even layer, layer: {layer}")
            for i in range(0, n - 1, 2):
              qml.CNOT(wires=[wires[i], wires[(i + 1)]])
            for i in range(1, n - 1, 2):
              qml.CNOT(wires=[wires[i], wires[(i + 1)]])
          elif id_init_circ and (layer % 2 == 1):
            print_cust(f"block_variational_circuit, brick, id_init_circ, odd layer, layer: {layer}")
            for i in reversed(range(1, n - 1, 2)):
              qml.CNOT(wires=[wires[i], wires[(i + 1)]])
            for i in reversed(range(0, n - 1, 2)):
              qml.CNOT(wires=[wires[i], wires[(i + 1)]])
        elif circuit_type == "x_shape":
          if not id_init_circ or (layer % 2 == 0):
            print_cust(f"block_variational_circuit, x_shape, not id_init_circ, even layer, layer: {layer}")
            for i in range(n - 1):
                control = wires[i]
                target = wires[(i + 1)]
                qml.CNOT(wires=[control, target])
                ending_control_idx = n - 2 - i
                if ending_control_idx != i:
                  control = wires[ending_control_idx]
                  target = wires[(ending_control_idx + 1)]
                  qml.CNOT(wires=[control, target])
          elif id_init_circ and (layer % 2 == 1):
            print_cust(f"block_variational_circuit, x_shape, id_init_circ, odd layer, layer: {layer}")
            for i in reversed(range(n - 1)):
                control = wires[i]
                target = wires[(i + 1)]
                qml.CNOT(wires=[control, target])
                ending_control_idx = n - 2 - i
                if ending_control_idx != i:
                  control = wires[ending_control_idx]
                  target = wires[(ending_control_idx + 1)]
                  qml.CNOT(wires=[control, target])
        elif circuit_type == "reversed_staircase":
          if not id_init_circ or (layer % 2 == 0):
            print_cust(f"block_variational_circuit, reversed_staircase, not id_init_circ, even layer, layer: {layer}")
            for i in range(n - 1, -1, -1):
                control = wires[i]
                target = wires[(i-1) % num_qubits]
                # TOMODIFY, depthFL: don't condition based on target QUBIT identifier, condition based on index
                if target == (n - 1):
                  if not generative and not is_hea:
                    qml.CNOT(wires=[control, target])
                else:
                  qml.CNOT(wires=[control, target])
          elif id_init_circ and (layer % 2 == 1):
            print_cust(f"block_variational_circuit, reversed_staircase, id_init_circ, odd layer, layer: {layer}")
            for i in range(n - 1):
              control = wires[i]
              target = wires[(i-1) % num_qubits]
              # TOMODIFY, depthFL: don't condition based on target QUBIT identifier, condition based on index
              if target == (n - 1):
                if not generative and not is_hea:
                  qml.CNOT(wires=[control, target])
              else:
                qml.CNOT(wires=[control, target])
        elif circuit_type == "v_shape":
          print_cust(f"block_variational_circuit, v_shape, layer, layer: {layer}")
          print_cust(f"block_variational_circuit, n: {n}, num_qubits: {num_qubits}")
          # TODO, depthFL: implement ID-init circ for v-shape circuit, too.
          # TOMODIFY, depthFL: this should go from 0 to n
          for i in range(n-1):
              print_cust(f"block_variational_circuit, v_shape, first loop, i: {i}")
              control = wires[i]
              target = wires[(i+1) % num_qubits]
              if i == (n - 1):
                if not generative and not is_hea:
                  qml.CNOT(wires=[control, target])
              else:
                qml.CNOT(wires=[control, target])
          for i in range(n - 1, -1, -1):
            print_cust(f"block_variational_circuit, v_shape, second loop, i: {i}")
            control = wires[i]
            target = wires[(i-1) % num_qubits]
            if i == 0:
              if not generative and not is_hea:
                qml.CNOT(wires=[control, target])
            else:
              qml.CNOT(wires=[control, target])
        elif circuit_type == "revstair_vshape":
          print_cust(f"block_variational_circuit, layer: {layer}")
          # if layers == 1:
          layer = offset_idx + layer
          print_cust(f"block_variational_circuit, after layer counts check, layer: {layer}")
          if (layer % 2 == 0):
            print_cust(f"block_variational_circuit, revstair_vshape, even layer, layer: {layer}")
            for i in range(n - 1, -1, -1):
                control = wires[i]
                target = wires[(i-1) % num_qubits]
                # TOMODIFY, depthFL: don't condition based on target QUBIT identifier, condition based on index
                if target == (n - 1):
                  if not generative and not is_hea:
                    qml.CNOT(wires=[control, target])
                else:
                  qml.CNOT(wires=[control, target])
          elif (layer % 2 == 1):
            print_cust(f"block_variational_circuit, revstair_vshape, odd layer, layer: {layer}")
            for i in range(n-1):
                control = wires[i]
                target = wires[(i+1) % num_qubits]
                if i == (n - 1):
                  if not generative and not is_hea:
                    qml.CNOT(wires=[control, target])
                else:
                  qml.CNOT(wires=[control, target])
            for i in range(n - 1, -1, -1):
              control = wires[i]
              target = wires[(i-1) % num_qubits]
              # TOMODIFY, depthFL: don't condition based on target QUBIT identifier, condition based on index
              if target == (n - 1):
                if not generative and not is_hea:
                  qml.CNOT(wires=[control, target])
              else:
                qml.CNOT(wires=[control, target])
        if inv_reupload_func is not None:
           print_cust(f"block_variational_circuit, applying inv_reupload_func")
           inv_reupload_func()