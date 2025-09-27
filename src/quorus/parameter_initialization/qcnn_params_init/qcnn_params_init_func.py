"""## Dynamic QCNN params initialization"""

from pennylane import numpy as np
from quorus.logging.custom_slog import print_cust
from quorus.parameter_initialization.identity_init_utils.id_init_func import make_paired_layers
import torch

def init_dynamic_qcnn_params(n_qubits, conv_layers, debug=False, zeros_init=False,
                             qubits_and_layers_to_add_block_params=[], generative=False, use_torch=False, alt_zeros_init=""):
    """
    Initialize dynamic QCNN parameters for a QCNN with a given number of convolution layers.

    After applying the pooling layers (one per conv_layer), compute the final number of qubits (current_qubits)
    remaining. Then, generate a final set of convolution layer parameters with the same format as the prior conv layers:
      - If current_qubits > 1:
           even_params has shape (current_qubits // 2, 12)
           odd_params  has shape ((current_qubits - 1) // 2, 12)
         and final_params is a tuple: (even_params, odd_params)
      - If current_qubits == 1:
           final_params is generated as a single-qubit rotation with shape (1, 3)

    Parameters:
      n_qubits: Integer representing the number of qubits in the circuit.
      conv_layers: Integer representing the number of convolution (and, assumed to also be the number of pooling) layers in the circuit.
      debug: Boolean indicating whether or not to initialize parameters to some distinct value, for debugging.
      zeros_init: Boolean indicating whether or not to initialize all parameters to 0.
      qubits_and_layers_to_add_block_params: List of (n_qubits, n_layers) representing the size of the block variational circuit for each of those qubits and layers.

    Returns:
      A tuple containing:
        (conv_params_list, pool_params_list, final_pool_param, final_params, bias_param, block_params_list)
    """

    conv_params_list = []
    pool_params_list = []
    current_qubits = n_qubits

    # Initialize parameters for each convolution + pooling layer.
    for layer in range(conv_layers):
        num_even_pairs = current_qubits // 2
        num_odd_pairs = (current_qubits - 1) // 2

        if debug:
            even_params = np.empty((num_even_pairs, 12))
            for j in range(num_even_pairs):
                for k in range(12):
                    even_params[j, k] = layer * 1e6 + (j * 12 + k + 1)
            odd_params = np.empty((num_odd_pairs, 12))
            for j in range(num_odd_pairs):
                for k in range(12):
                    odd_params[j, k] = layer * 1e6 + (j * 12 + k + 1)
        else:
            even_params = (np.zeros((num_even_pairs, 12)) if zeros_init
                           else np.random.randn(num_even_pairs, 12))
            odd_params = (np.zeros((num_odd_pairs, 12)) if zeros_init
                          else np.random.randn(num_odd_pairs, 12))

        conv_params_list.append((even_params, odd_params))

        if debug:
            pool_params = np.empty((num_even_pairs, 1))
            for j in range(num_even_pairs):
                pool_params[j, 0] = layer * 1e6 + (j + 1)
        else:
            pool_params = np.random.randn(num_even_pairs, 1)
        pool_params_list.append(pool_params)

        # Update current_qubits based on pooling logic.
        # When the number of qubits is even: new number = n/2.
        # When odd: new number = (n//2) + 1 (keeping the last unpaired qubit).
        if current_qubits % 2 == 0:
            current_qubits = current_qubits // 2
        else:
            current_qubits = current_qubits // 2 + 1

    # After conv_layers and pooling, initialize final conv layer parameters based on remaining qubits.
    if current_qubits > 1 and not generative:
        num_even_pairs_final = current_qubits // 2
        num_odd_pairs_final = (current_qubits - 1) // 2
        if debug:
            final_even = np.empty((num_even_pairs_final, 12))
            final_odd = np.empty((num_odd_pairs_final, 12))
            for j in range(num_even_pairs_final):
                for k in range(12):
                    final_even[j, k] = conv_layers * 1e6 + (j * 12 + k + 1)
            for j in range(num_odd_pairs_final):
                for k in range(12):
                    final_odd[j, k] = conv_layers * 1e6 + (j * 12 + k + 1)
        else:
            final_even = (np.random.randn(num_even_pairs_final, 12)
                          if not zeros_init else np.zeros((num_even_pairs_final, 12)))
            final_odd = (np.random.randn(num_odd_pairs_final, 12)
                         if not zeros_init else np.zeros((num_odd_pairs_final, 12)))
        final_params = (final_even, final_odd)
    else:
        # If only one qubit remains, generate parameters for a single-qubit rotation.
        if debug:
            final_params = np.array([[conv_layers * 1e6 + 1, conv_layers * 1e6 + 2, conv_layers * 1e6 + 3]])
        else:
            final_params = (np.random.randn(1, 3) if not zeros_init else np.zeros((1, 3)))

    # Generate final_pool_param and bias_param as before.
    if debug:
        final_pool_param = np.array([conv_layers * 1e6 + 1])
        bias_param = np.array([conv_layers * 1e6 + 1])
    else:
        final_pool_param = np.random.randn(1)
        bias_param = np.random.randn(1)

    # Initialize block parameters.
    block_params_list = []
    for block_idx, (num_qubits_bp, num_layers_bp) in enumerate(qubits_and_layers_to_add_block_params):
        shape = (num_layers_bp, num_qubits_bp, 3)
        if not use_torch:
          bp = np.zeros(shape) if zeros_init or debug else np.random.randn(*shape)
          # NOTE, layers: this is an override of bp.
          if zeros_init and alt_zeros_init == "posneg":
            print_cust(f"init_dynamic_qcnn_params, posneg numpy initialization")
            bp = make_paired_layers(shape, "numpy")
          elif zeros_init and alt_zeros_init == "random":
            print_cust(f"init_dynamic_qcnn_params, random numpy initialization")
            bp = np.random.randn(*shape)
        else:
          bp = torch.zeros(shape) if zeros_init or debug else torch.randn(*shape)
          if zeros_init and alt_zeros_init == "posneg":
            print_cust(f"init_dynamic_qcnn_params, posneg torch initialization")
            bp = make_paired_layers(shape, "torch", requires_grad=True)
          elif zeros_init and alt_zeros_init == "random":
            print_cust(f"init_dynamic_qcnn_params, random torch initialization")
            bp = torch.randn(*shape)
        block_params_list.append(bp)

    return (tuple(conv_params_list), tuple(pool_params_list), final_pool_param,
            final_params, bias_param, block_params_list)