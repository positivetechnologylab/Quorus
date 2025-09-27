import pennylane as qml
from quorus.metrics_funcs.pennylane_lossfns.pennylane_lossfn import pennylane_loss_fn

def compute_loss_angle_param_batch(params, X_angles, y, layers=1, shots=1024, batch_size=None, qnode=None):
    """
    Parameters:
      params: List of parameters to feed into the quantum model.
      X_angles: Numpy array of input data to feed into the model
      y: Expected classification outputs for each of the input datum (assumed to be integers representing the class of interest).
      layers: Integer representing the number of layers in the QCNN.
      batch_size: Integer representing the batch size.
      shots: Integer representing the number of shots used to evaluate the QCNN.
      qnode: A Pennylane QNode object for evaluating the quantum model.

    Computes the loss for all of the input data specified.

    Returns:
      A float representing the average cross entropy loss over the dataset.
    """
    # total_loss = 0
    # for xi, yi in zip(X_angles, y):
    #     total_loss += pennylane_loss_fn(params, xi, yi, qnode)
    # return total_loss / len(X_angles)
    losses = [pennylane_loss_fn(params, xi, yi, qnode) for xi, yi in zip(X_angles, y)]
    return qml.math.mean(qml.math.stack(losses))