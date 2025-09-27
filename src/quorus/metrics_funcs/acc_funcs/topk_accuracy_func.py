from pennylane import numpy as np

"""## Top K accuracy"""

def compute_top_k_acc_angle_param_batch(params, X_angles, y, layers=1, shots=1024, batch_size=None, qnode=None):
    """
    Computes an array of top-k accuracies for a multiclass classifier.

    For each k = 1, 2, ..., N (where N is the total number of classes), the function
    checks how many times the true label appears in the top k predictions (sorted in descending
    order) produced by the qnode for the provided inputs, and then returns the proportion of
    correct cases for each k.

    Parameters:
        params: Parameters to be passed to the qnode.
        X_angles: Array-like, inputs to the qnode function.
        y: Array-like, true labels.
        layers: (Optional) Parameter for potential layer configuration (currently unused).
        shots: (Optional) Parameter for simulation shots (currently unused).
        batch_size: (Optional) Parameter for batch processing (currently unused).
        qnode: A function that takes (input_angles, params) and returns a vector of probabilities.

    Returns:
        top_k_accuracies: A NumPy array of shape (N,) where the element at index k-1
                          is the top-k accuracy (with k from 1 to N).
    """
    # Compute the number of classes using the first qnode call (assumes X_angles is non-empty)
    first_probs = qnode(X_angles[0], params)
    num_classes = len(first_probs)

    top_k_counts = np.zeros(num_classes)
    total_samples = len(X_angles)

    # Loop over each input and its corresponding true label.
    for i, input_angles in enumerate(X_angles):
        probs = qnode(input_angles, params)
        # Get indices of predictions sorted from highest to lowest probability.
        sorted_indices = np.argsort(-np.array(probs))
        true_label = y[i]

        # For each k from 1 to the total number of classes, check if the true label is among the top k predictions.
        for k in range(1, num_classes + 1):
            if true_label in sorted_indices[:k]:
                top_k_counts[k - 1] += 1

    # Convert counts to accuracies
    top_k_accuracies = top_k_counts / total_samples
    return top_k_accuracies