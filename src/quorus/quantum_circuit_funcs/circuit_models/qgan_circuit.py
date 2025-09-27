from quorus.logging.custom_slog import print_cust
import pennylane as qml
from quorus.quantum_circuit_funcs.data_encoding.data_encoding_funcs import pennylane_angleencode
from quorus.quantum_circuit_funcs.circuit_models_components.variational_layers_func import block_variational_circuit

"""## QGAN Circuit"""

def QGAN_circuit(noise, n_qubits, block_params_list=[]):
  block_params_list_sorted = sorted(block_params_list, key=lambda x:x.shape[1])
  for block_params in block_params_list_sorted:
    print_cust(f"QGAN_circuit, block_params.shape: {block_params.shape}")

  if len(noise.shape) == 1:
    pennylane_angleencode(range(0, n_qubits), noise, generative=True)
  elif len(noise.shape) == 2:
    print_cust(f"QGAN_circuit, running pennylane.angleencode")
    qml.templates.AngleEmbedding(noise, wires=range(0, n_qubits), rotation="X")
    qml.templates.AngleEmbedding(noise, wires=range(0, n_qubits), rotation="Y")
  wires = list(range(n_qubits))

  for block_params in block_params_list_sorted:
    num_layers_bp = block_params.shape[0]
    num_qubits_bp = block_params.shape[1]
    block_variational_circuit(block_params, num_layers_bp, list(range(num_qubits_bp)), generative=True)

  return [qml.expval(qml.Z(wires=qubit_idx_ret)) for qubit_idx_ret in range(n_qubits)]