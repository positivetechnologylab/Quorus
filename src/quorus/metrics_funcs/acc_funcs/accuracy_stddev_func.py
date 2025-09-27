from pennylane import numpy as np

"""## Standard deviation of accuracy function"""

def compute_std_acc_angle_param_batch(params, X_angles, y, layers=1, shots=1024, batch_size=None, qnode=None):
    """
    Computes the standard deviation of per-class accuracies for a multiclass classifier.

    For each class, the function computes the accuracy (i.e., the fraction of correct predictions
    for examples belonging to that class) and then calculates the standard deviation of these accuracies.

    Parameters:
        params: Parameters to be passed to the qnode.
        X_angles: Array-like, inputs to the qnode function.
        y: Array-like, true labels.
        layers: (Optional) Parameter for potential layer configuration (currently unused).
        shots: (Optional) Parameter for simulation shots (currently unused).
        batch_size: (Optional) Parameter for batch processing (currently unused).
        qnode: A function that takes (input_angles, params) and returns a vector of probabilities.

    Returns:
        std_acc: The standard deviation of the per-class accuracies.
    """
    preds = []
    # Generate predictions using the qnode
    for input_angles in X_angles:
        probs = qnode(input_angles, params)
        preds.append(np.argmax(probs))

    preds = np.array(preds)

    # Compute the accuracy for each class.
    unique_classes = np.unique(y)
    class_accuracies = []

    for cls in unique_classes:
        # Indices for the current class
        indices = np.where(y == cls)[0]
        # Compute accuracy for the current class
        if len(indices) > 0:
            acc = np.mean(preds[indices] == y[indices])
            class_accuracies.append(acc)
        else:
            # If no examples are present for a class, we can either ignore or set a default accuracy (e.g., 0.0).
            class_accuracies.append(0.0)

    # Calculate the standard deviation across the per-class accuracies.
    std_acc = np.std(class_accuracies)
    return std_acc