"""## Helper for Alternate Zeros Init for Dynamic QCNN Params"""

from typing import Tuple, Optional, Literal
from pennylane import numpy as np
from quorus.logging.custom_slog import print_cust
import torch

def make_paired_layers(
    shape: Tuple[int, int, int],
    backend: Literal["numpy", "torch"] = "numpy",
    *,
    seed: Optional[int] = None,
    dtype=None,                 # np.dtype or torch.dtype (optional)
    device: Optional[str] = None,  # torch device string (e.g., "cuda") if backend="torch"
    requires_grad: bool = False     # only used for torch
):
    """
    Generate an array/tensor of shape (num_layers, num_qubits, 3).

    Behavior:
      - If num_layers is a multiple of 3: build triads (params, 0, -params) per group of 3.
      - Else if num_layers is a multiple of 2: use pairs as before (params, -params) per group of 2.
      - Otherwise: use the original pair logic and append leftover random layers.

    Notes:
      - When num_layers is divisible by both 2 and 3, triads take precedence.
    """
    L, Q, C = shape
    print_cust(f"make_paired_layers, shape: {shape}")
    if C != 3:
        raise ValueError(f"shape[-1] must be 3, got {C}")
    if L < 0 or Q < 0:
        raise ValueError("num_layers and num_qubits must be non-negative")

    # Decide mode (triads first to give it precedence over pairs)
    use_triads = (L % 3 == 0 and L > 0)
    use_pairs = (L % 2 == 0 and not use_triads and L > 0)

    if backend == "numpy":

        # if seed is not None:
        #     np.random.seed(seed)

        if use_triads:
            # --- TRIADS: (params, 0, -params) ---
            t = L // 3
            if t > 0:
                base = np.random.randn(t, 1, Q, 3)                         # (t,1,Q,3)
                if dtype is not None:
                    base = base.astype(dtype)
                zeros = np.zeros_like(base)                                 # (t,1,Q,3)
                triad = np.concatenate([base, zeros, -base], axis=1)       # (t,3,Q,3)
                stacked = triad.reshape(3 * t, Q, 3)                        # (3t,Q,3)
            else:
                stacked = np.empty((0, Q, 3), dtype=dtype if dtype is not None else float)
            return stacked

        # Fallback to PAIRS (original behavior, including leftovers)
        k = L // 2  # number of full pairs
        if k > 0:
            base = np.random.randn(k, 1, Q, 3)                              # (k,1,Q,3)
            if dtype is not None:
                base = base.astype(dtype)
            pair = np.concatenate([base, -base], axis=1)                    # (k,2,Q,3)
            stacked = pair.reshape(2 * k, Q, 3)                             # (2k,Q,3)
        else:
            stacked = np.empty((0, Q, 3), dtype=dtype if dtype is not None else float)

        if (L % 2) == 1:  # odd layer count → add one extra layer
            extra = np.random.randn(1, Q, 3)
            if dtype is not None:
                extra = extra.astype(dtype)
            return np.concatenate([stacked, extra], axis=0)
        return stacked

    elif backend == "torch":
        # import torch

        # if seed is not None:
        #     torch.manual_seed(seed)

        final_dtype = dtype if dtype is not None else torch.get_default_dtype()

        if use_triads:
            # --- TRIADS: (params, 0, -params) ---
            t = L // 3
            if t > 0:
                base = torch.randn((t, 1, Q, 3), dtype=final_dtype, device=device,
                                   requires_grad=requires_grad)             # (t,1,Q,3)
                zeros = torch.zeros_like(base)                               # (t,1,Q,3)
                triad = torch.cat([base, zeros, -base], dim=1)              # (t,3,Q,3)
                stacked = triad.reshape(3 * t, Q, 3)                         # (3t,Q,3)
            else:
                stacked = torch.empty((0, Q, 3), dtype=final_dtype, device=device)
                if requires_grad:
                    stacked.requires_grad_()
            return stacked

        # Fallback to PAIRS (original behavior, including leftovers)
        k = L // 2
        if k > 0:
            base = torch.randn((k, 1, Q, 3), dtype=final_dtype, device=device,
                               requires_grad=requires_grad)                  # (k,1,Q,3)
            pair = torch.cat([base, -base], dim=1)                           # (k,2,Q,3)
            stacked = pair.reshape(2 * k, Q, 3)                              # (2k,Q,3)
        else:
            stacked = torch.empty((0, Q, 3), dtype=final_dtype, device=device)
            if requires_grad:
                stacked.requires_grad_()

        if (L % 2) == 1:
            extra = torch.randn((1, Q, 3), dtype=final_dtype, device=device,
                                requires_grad=requires_grad)
            return torch.cat([stacked, extra], dim=0)
        return stacked

    else:
        raise ValueError('backend must be either "numpy" or "torch"')