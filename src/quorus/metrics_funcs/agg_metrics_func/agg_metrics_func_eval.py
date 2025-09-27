from pennylane import numpy as np
import torch
from quorus.logging.custom_slog import print_cust
import pennylane as qml

"""## Aggregated metrics function"""

def compute_metrics_angle_param_batch(params, X_angles, y, layers=1, shots=1024, batch_size=None, qnode=None, math_int=np):
    """
    Function that computes the average accuracies, standard deviation of the per-class accuracies, top-k accuracy, and average loss
    with just one call of the quantum circuit.

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
    if math_int == np:
      container_creator = np.array
      float_dtype = np.float64
    elif math_int == torch:
      container_creator = torch.tensor
      float_dtype = torch.float32
    print_cust(f"compute_metrics_angle_param_batch, math_int: {math_int}")
    print_cust(f"compute_metrics_angle_param_batch, container_creator: {container_creator}")
    # Compute the number of classes using the first qnode call (assumes X_angles is non-empty)
    # DONE: TOMODIFY, depthFL: this is multiple output probabilities. you need to ensemble it, and/or compute the accs for EACH classifier model.
    first_probs = qnode(X_angles[0], params)
    if len(first_probs.shape) == 2:
      # then assumed to be of shape (n_classifiers, prob_dim)
      print_cust(f"compute_metrics_angle_param_batch, len(first_probs.shape) == 2, first_probs.shape: {first_probs.shape}")
      num_classes = first_probs.shape[1]
    else:
      # should be of shape prob_dim
      print_cust(f"compute_metrics_angle_param_batch, len(first_probs.shape) != 2, first_probs.shape: {first_probs.shape}")
      num_classes = len(first_probs)


    top_k_counts = math_int.zeros(num_classes)
    total_samples = len(X_angles)

    preds = []
    losses = []

    epsilon = 1e-7

    # Loop over each input and its corresponding true label.
    # DONE: TOMODIFY, depthFL: this is multiple output probabilities. you need to ensemble it, and/or compute the accs for EACH classifier model.
    all_probs = qnode(X_angles, params)
    gen_all_probs = None
    if len(all_probs.shape) == 3:
      print_cust(f"compute_metrics_angle_param_batch, len(all_probs.shape) == 3, all_probs.shape: {all_probs.shape}")
      # should be of shape (n_classifiers, input_dim, prob_dim)
      # TOMODIFY, depthFL: BEFORE taking the mean, get the statistics for each INDIVIDUAL classifier; this could be helpful.
      gen_all_probs = all_probs
      all_probs = all_probs.mean(axis=0)
    print_cust(f"compute_metrics_angle_param_batch, all_probs.shape: {all_probs.shape}")
    # of shape input_dim, prob_dim
    for i, probs in enumerate(all_probs):
        # probs = qnode(input_angles, params)
        preds.append(math_int.argmax(probs))
        # Get indices of predictions sorted from highest to lowest probability.

        sorted_indices = math_int.argsort(-container_creator(probs))
        true_label = y[i]

        loss = -qml.math.log(probs[int(true_label)] + epsilon)
        losses.append(loss)

        # For each k from 1 to the total number of classes, check if the true label is among the top k predictions.
        for k in range(1, num_classes + 1):
            if true_label in sorted_indices[:k]:
                top_k_counts[k - 1] += 1

    # Convert counts to accuracies
    top_k_accuracies = top_k_counts / total_samples


    preds = container_creator(preds)

    # Compute the accuracy for each class.
    # CHANGED, layers: wrapped y in container_creator b/c torch.unique only takes in tensors; make sure that functionality is still OK in the numpy case.
    unique_classes = math_int.unique(container_creator(y))
    class_accuracies = []

    for cls in unique_classes:
        # Indices for the current class
        indices = math_int.where(container_creator(y) == cls)[0]
        # Compute accuracy for the current class
        if len(indices) > 0:
            acc = math_int.mean(preds[indices] == container_creator(y)[indices], dtype=float_dtype)
            class_accuracies.append(acc)
        else:
            # If no examples are present for a class, we can either ignore or set a default accuracy (e.g., 0.0).
            class_accuracies.append(0.0)

    # Calculate the standard deviation across the per-class accuracies.
    # NOTE, layers (although this is not specific to the layer classifier setup): for torch, ddof=1 default; for numpy, ddof=0 default, so will get diff stddevs. (prob should use)
    # sample stddev here.
    std_acc = math_int.std(container_creator(class_accuracies))

    avg_acc = math_int.mean(preds == container_creator(y), dtype=float_dtype)

    avg_loss = qml.math.mean(qml.math.stack(losses))

    if gen_all_probs is not None:
      avg_acc_classifiers = []
      std_acc_classifiers = []
      top_k_accuracies_classifiers = []
      avg_loss_classifiers = []
      print_cust(f"compute_metrics_angle_param_batch, gen_all_probs is not None")
      # gen_all_probs assumed to be of shape 3
      for all_probs in gen_all_probs:
        # TOMODIFY, depthFL: encapsulate this logic for computing classifier accuracies in a function.
        top_k_counts = math_int.zeros(num_classes)
        total_samples = len(X_angles)

        preds = []
        losses = []

        epsilon = 1e-7

        print_cust(f"compute_metrics_angle_param_batch, all_probs.shape: {all_probs.shape}")
        # of shape input_dim, prob_dim
        for i, probs in enumerate(all_probs):
            # probs = qnode(input_angles, params)
            preds.append(math_int.argmax(probs))
            # Get indices of predictions sorted from highest to lowest probability.

            sorted_indices = math_int.argsort(-container_creator(probs))
            true_label = y[i]

            loss = -qml.math.log(probs[int(true_label)] + epsilon)
            losses.append(loss)

            # For each k from 1 to the total number of classes, check if the true label is among the top k predictions.
            for k in range(1, num_classes + 1):
                if true_label in sorted_indices[:k]:
                    top_k_counts[k - 1] += 1

        # Convert counts to accuracies
        top_k_accuracies_subclassif = top_k_counts / total_samples


        preds = container_creator(preds)

        # Compute the accuracy for each class.
        # CHANGED, layers: wrapped y in container_creator b/c torch.unique only takes in tensors; make sure that functionality is still OK in the numpy case.
        unique_classes = math_int.unique(container_creator(y))
        class_accuracies = []

        for cls in unique_classes:
            # Indices for the current class
            indices = math_int.where(container_creator(y) == cls)[0]
            # Compute accuracy for the current class
            if len(indices) > 0:
                acc = math_int.mean(preds[indices] == container_creator(y)[indices], dtype=float_dtype)
                class_accuracies.append(acc)
            else:
                # If no examples are present for a class, we can either ignore or set a default accuracy (e.g., 0.0).
                class_accuracies.append(0.0)

        # Calculate the standard deviation across the per-class accuracies.
        # NOTE, layers (although this is not specific to the layer classifier setup): for torch, ddof=1 default; for numpy, ddof=0 default, so will get diff stddevs. (prob should use)
        # sample stddev here.
        std_acc_subclassif = math_int.std(container_creator(class_accuracies))

        avg_acc_subclassif = math_int.mean(preds == container_creator(y), dtype=float_dtype)

        avg_loss_subclassif = qml.math.mean(qml.math.stack(losses))

        avg_acc_classifiers.append(avg_acc_subclassif)
        std_acc_classifiers.append(std_acc_subclassif)
        top_k_accuracies_classifiers.append(top_k_accuracies_subclassif)
        avg_loss_classifiers.append(avg_loss_subclassif)




    if gen_all_probs is None:
      return avg_acc, std_acc, top_k_accuracies, avg_loss
    else:
      print_cust(f"compute_metrics_angle_param_batch, gen_all_probs is not None, returning")
      return avg_acc, std_acc, top_k_accuracies, avg_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs