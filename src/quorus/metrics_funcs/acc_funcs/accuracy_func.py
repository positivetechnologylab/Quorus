from pennylane import numpy as np

"""## Accuracy computation function"""

def compute_avg_acc_angle_param_batch(params, X_angles, y, layers=1, shots=1024, batch_size=None, qnode=None):
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
      A float representing the average cross entropy accuracy over the dataset.

    Computes average accuracy by comparing the index of the maximum probability
    in the output vector to the true label.
    """
    preds = []
    for input_angles in X_angles:
        probs = qnode(input_angles, params)
        preds.append(np.argmax(probs))
    preds = np.array(preds)
    return np.mean(preds == y)