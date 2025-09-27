"""# Angle encoding evaluation funcs

## Pennylane loss function
"""

import pennylane as qml

def pennylane_loss_fn(params, input_angles, y, qnode):
    """
    Parameters:
      params: List of parameters to feed into the quantum model.
      input_angles: Input data to feed into the quantum model.
      y: Expected classification output (assumed to be an integer representing the class of interest).
      qnode: A Pennylane QNode object for evaluating the quantum model.


    Compute the multi-class cross entropy loss.

    The QCNN now returns a probability vector.
    y is expected to be an integer label.

    Returns:
      A float representing the multi class cross entropy loss.
    """
    probs = qnode(input_angles, params)
    epsilon = 1e-7
    # For the given true label y, compute negative log-likelihood
    return -qml.math.log(probs[int(y)] + epsilon)