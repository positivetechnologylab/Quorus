from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

def tuples_allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Parameters:
      a: a list, tuple, or numpy array composed of other lists, tuples or numpy arrays
      b: a list, tuple, or numpy array composed of other lists, tuples or numpy arrays
      rtol: float representing the relative tolerance parameter
      atol: float representing the absolute tolerance parameter
      equal_nan: boolean indicating whether to compare NaN's as equal.
    """
    # If both are list or tuple, compare elementwise recursively.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(tuples_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
                   for x, y in zip(a, b))
    else:
        # Otherwise, assume they are arrays or scalars.
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)