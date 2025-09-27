"""## Helper Function for Computing Param Diffs"""

import torch
from typing import Iterable, Dict, Optional

def l2_state_dict_difference(
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    *,
    keys: Optional[Iterable[str]] = None,
    strict: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    include_buffers: bool = True
) -> torch.Tensor:
    """
    Compute the L2 norm of the difference between two PyTorch state dicts.

    Parameters
    ----------
    state_dict_a, state_dict_b :
        The two state dicts (e.g. model.state_dict()) whose parameter/buffer
        tensors you want to compare.
    keys :
        Optional iterable of keys to restrict the comparison. If None, will
        use the intersection of keys (parameters +, optionally, buffers).
    strict :
        If True, raise an error when a key in `keys` is missing from either
        dict. If False, silently skip missing keys.
    device :
        Device to perform the accumulation on. If None, uses the device of
        each individual tensor (differences are accumulated on the first
        encountered device, moving others as needed).
    dtype :
        Optional dtype to cast tensors before differencing (e.g. torch.float32
        for numerical stability / consistency).
    include_buffers :
        If False, attempts to skip typical non-parameter buffers (by a simple
        heuristic: keys containing 'running_' or 'num_batches_tracked').

    Returns
    -------
    torch.Tensor
        A scalar tensor: sqrt( sum_i || param_i^A - param_i^B ||_2^2 ).
        (Equivalent to the global L2 norm of all flattened differences.)

    Notes
    -----
    * Implemented in a streaming fashion to avoid allocating one huge
      concatenated tensor—memory efficient for large models.
    * Uses manual accumulation + sqrt instead of the (deprecated) torch.norm
      over a concatenated tensor; could also use torch.linalg.norm on a final
      flattened difference vector, but that would require materializing it.
    """
    if keys is None:
        keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    else:
        keys = list(keys)

    # Optionally filter out common buffer keys
    if not include_buffers:
        def is_buffer(k: str) -> bool:
            # Heuristic; tailor as needed
            return ('running_' in k) or ('_tracked' in k)
        keys = [k for k in keys if not is_buffer(k)]

    total_ssd = None
    target_device = device
    target_dtype = dtype

    for k in keys:
        if k not in state_dict_a or k not in state_dict_b:
            if strict:
                raise KeyError(f"Key '{k}' missing from one of the state dicts.")
            else:
                continue

        ta = state_dict_a[k]
        tb = state_dict_b[k]
        if ta.shape != tb.shape:
            if strict:
                raise ValueError(f"Shape mismatch for key '{k}': {ta.shape} vs {tb.shape}")
            else:
                continue

        # Decide accumulation device/dtype lazily
        if target_device is None:
            target_device = ta.device
        if target_dtype is None:
            target_dtype = ta.dtype

        diff = (ta.to(device=target_device, dtype=target_dtype) -
                tb.to(device=target_device, dtype=target_dtype))
        ssd = diff.pow(2).sum()

        if total_ssd is None:
            total_ssd = ssd
        else:
            total_ssd += ssd

    if total_ssd is None:
        # No comparable keys
        return torch.tensor(0.0, device=device or 'cpu', dtype=dtype or torch.float32)
    return total_ssd.sqrt()