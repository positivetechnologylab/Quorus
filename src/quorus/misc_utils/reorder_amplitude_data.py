from pennylane import numpy as np
from quorus.logging.custom_slog import print_cust

"""## Amplitude Embedding Data Reordering Function"""

def reorder_amplitude_data(data: np.ndarray, qubit_order: list[int]) -> np.ndarray:
    """
    Reorder (and, if needed, zero-pad or truncate) amplitude-embedded data
    for a new qubit ordering.

    Args:
        data (np.ndarray): Array of shape (n_samples, N_old), where N_old = 2**k_old.
        qubit_order (list[int]): New ordering of qubit indices of length k_new;
                                 qubit_order[i] is the original qubit index
                                 to place at position i.

    Returns:
        np.ndarray: Array of shape (n_samples, 2**k_new), with amplitudes:
            - permuted so that the i-th qubit in the new embedding
              corresponds to the original qubit_order[i],
            - zero-padded if N_old < 2**k_new,
            - truncated if N_old > 2**k_new.
    """
    # NOTE, layers, ampembed: be careful about how this function is called. make sure that the inputs to the
    # calling function here fit; ideally, I don't want to call this function.
    print_cust(f"reorder_amplitude_data, data.shape: {data.shape}")
    print_cust(f"reorder_amplitude_data, qubit_order: {qubit_order}")
    n_samples, old_size = data.shape

    # --- sanity-check old size is a power of 2 ---
    # old_nq = int(np.log2(old_size))
    # if 2**old_nq != old_size:
    #     raise ValueError("`data` must have 2^k columns")

    # --- compute new dimensions ---
    new_nq   = len(qubit_order)
    new_size = 1 << new_nq

    # --- truncate or pad as needed ---
    if old_size > new_size:
        # too many features → truncate
        data = data[:, :new_size]
    elif old_size < new_size:
        # too few features → pad with zeros
        padded = np.zeros((n_samples, new_size), dtype=data.dtype)
        padded[:, :old_size] = data
        data = padded
    # now data.shape[1] == new_size

    print_cust(f"reorder_amplitude_data, data.shape: {data.shape}")

    # --- build the bit-permutation map ---
    perm = np.empty(new_size, dtype=int)
    for j in range(new_size):
        bits     = (j >> np.arange(new_nq)) & 1          # binary of j
        bits_new = bits[qubit_order]                     # permute bit positions
        perm[j]  = int(np.dot(bits_new, 1 << np.arange(new_nq)))  # reassemble

    # --- apply permutation across amplitudes ---
    return data[:, perm]