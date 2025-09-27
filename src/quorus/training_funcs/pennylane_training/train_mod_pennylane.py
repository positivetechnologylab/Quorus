"""## Core training function"""

import copy
from pennylane.optimize import AdamOptimizer
from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
import pennylane as qml
from quorus.metrics_funcs.pennylane_lossfns.pennylane_lossfn_batch import compute_loss_angle_param_batch
from quorus.param_processing.numpy_funcs.param_grads_utils_pennylane import tree_to_list, grad_l2_norm
from quorus.param_processing.numpy_funcs.param_comparison import tuples_allclose

# NOTE: this is NOT used in final quorus version; is here because previous iterations of this project used this func.
# so, no guarantees that this function SPECIFICALLY works as intended (particularly, if autograd/backprop yield the exact,
# expected, correct gradients)
def train_epochs_angle_param_adam(params, X_angles, y, X_val, y_val, n_epochs=5, layers=1, shots=1024, batch_size=32,
                                  lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                                  grad_func=None, trainable_mask=None, qnode=None,
                                  patience=3):

    """
    Trains through n_epochs epochs over the given X_angles dataset in minibatches, using the validation set to have early termination.

    Parameters:
      params: a list of numpy arrays
      X_angles: Array-like, inputs to the qnode function.
      y: Array-like, true labels.
      X_val: Array-like, inputs to qnode function used for early termination.
      y_val: Array-like, true labels used for early termination.
      n_epochs: Integer, number of epochs used for training.
      layers: (Optional) Parameter for potential layer configuration (currently unused).
      shots: (Optional) Parameter for simulation shots (currently unused).
      batch_size: (Optional) Parameter for batch processing (currently unused).
      lr: Integer representing the learning rate for the optimizer.
      beta1: Integer representing the beta1 parameter for the optimizer (currently unused).
      beta2: Integer representing the beta2 parameter for the optimizer (currently unused).
      epsilon: Float representing some minimum precision (currently unused).
      grad_func: Function used to apply the gradient (currently unused; could be used in the future for more decoupling/separation of concerns).
      trainable_mask: List of numpy arrays used to specify what parameters to NOT apply the gradient on.
      qnode: A function that takes (input_angles, params) and returns a vector of probabilities.
      patience: Integer representing the maximum number of epochs that is allowed for validation loss to not increase.

    Returns:
      params: a list of numpy arrays representing the trained params
      minibatch_losses: a numpy array containing the losses for each minibatch
      validation_losses: a numpy array containing the losses on the validation set
    """
    # Initialize containers to store metrics as well as the optimizer
    n_samples = X_angles.shape[0]
    minibatch_losses = []
    validation_losses = []
    opt = AdamOptimizer(stepsize=lr)
    best_val_loss = float('inf')
    patience_count = 0
    # print_cust(f"train_epochs_angle_param_adam, params[0].shape: {params[0].shape}")

    # Record the original nested structure as a template
    structure_template = params

    print_cust(f"train_epochs_angle_param_adam, n_samples: {n_samples}")
    # For each epoch, train on randomly sampled minibatches.
    for epoch in range(n_epochs):
        print_cust(f"[Adam SGD] Epoch {epoch+1}/{n_epochs}")
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            X_batch = X_angles[batch_indices]
            y_batch = y[batch_indices]
            old_params = copy.deepcopy(params)

            # Compute gradient on the mini-batch
            grads = qml.grad(compute_loss_angle_param_batch, argnum=0)(
                params, X_batch, y_batch, layers=layers, shots=shots, batch_size=len(X_batch), qnode=qnode
            )

            # --------- flatten *one* level ----------
            flat_p,  rebuild = tree_to_list(params)
            flat_g, _        = tree_to_list(grads)       # identical structure ⇒ same rebuild

            # ensure differentiable type
            flat_p = [np.asarray(p, requires_grad=True) for p in flat_p]
            # print_cust(f"train_epochs_angle_param_adam, grads after grad calculation: {grads}")
            # grads = np.array(grads).flatten()
            # If a trainable mask is provided, apply it elementwise
            # if trainable_mask is not None:
            #     def apply_mask(grad, mask):
            #         """Element‑wise multiply `grad` by `mask`, preserving the nested structure.

            #         Both grad and mask must have *exactly* the same tree layout.
            #         Mask entries can be 0/1, booleans, or any broadcast‑compatible scalars.
            #         """
            #         # Same container (list vs tuple) is preserved with `type(grad)(...)`
            #         if isinstance(grad, (list, tuple)):
            #             if len(grad) != len(mask):
            #                 raise ValueError("Mask and gradient have different lengths")
            #             return type(grad)(apply_mask(g, m) for g, m in zip(grad, mask))
            #         else:
            #             # Convert to array once so broadcasting works and the dtypes match
            #             return grad * np.asarray(mask)
            #     grads = apply_mask(grads, trainable_mask)
            #     # print_cust(f"train_epochs_angle_param_adam, masked grads: {grads}")
            # (optional) mask
            if trainable_mask is not None:
                flat_mask, _ = tree_to_list(trainable_mask)
                # print_cust(f"train_epochs_angle_param_adam, flat_mask: {flat_mask}")
                # print_cust(f"train_epochs_angle_param_adam, in trainable_mask conditional, flat_g: {flat_g}")
                flat_g = [g * m for g, m in zip(flat_g, flat_mask)]
                # print_cust(f"train_epochs_angle_param_adam, in trainable_mask conditional, after applying mask, flat_g: {flat_g}")


            # print_cust(f"train_epochs_angle_param_adam, flat_g: {flat_g}")

            # print_cust(f"train_epochs_angle_param_adam, old_params: {old_params}")
            # if trainable_mask is not None:
            #   print_cust(f"train_epochs_angle_param_adam, flat_p: {flat_p}")

            # --------- Adam step ----------
            flat_p = opt.apply_grad(flat_g, flat_p)
            # if trainable_mask is not None:
            #   print_cust(f"train_epochs_angle_param_adam, flat_p after grad application: {flat_p}")

            # --------- put tensors back ----------
            params  = rebuild(flat_p)

            # print_cust(f"train_epochs_angle_param_adam, params after grad application: {params}")
            print_cust(f"train_epochs_angle_param_adam, grad_l2_norm(grads): {grad_l2_norm(flat_g)}")

            # Flatten parameters and gradients recursively
            # flat_params, shapes = flatten_params_recursive(params)
            # flat_grad = flatten_grad_recursive(grads)

            # print_cust(f"train_epochs_angle_param_adam, np.linalg.norm(flat_grad): {np.linalg.norm(flat_grad)}")
            # print_cust(f"train_epochs_angle_param_adam, flat_params.shape: {flat_params.shape}")
            # print_cust(f"train_epochs_angle_param_adam, flat_grad.shape: {flat_grad.shape}")
            # print_cust(f"train_epochs_angle_param_adam, shapes: {shapes}")

            # Convert flat_params to a Pennylane array with requires_grad=True
            # flat_params = np.array(flat_params, requires_grad=True)

            # Update parameters using the Adam optimizer
            # updated_flat_params = opt.apply_grad(flat_grad, flat_params)
            # updated_flat_params = np.array(updated_flat_params, requires_grad=True)

            # Reconstruct the nested structure of parameters
            # params = unflatten_params_recursive(updated_flat_params, shapes, structure_template)

            if tuples_allclose(old_params, params):
              print_cust(f"train_epochs_angle_param_adam, old params and params are basically the same after gradient application")
            # print_cust(f"train_epochs_angle_param_adam, grads after grad application: {grads}")
            # print_cust(f"train_epochs_angle_param_adam, params after grad application: {params}")
            loss_batch = compute_loss_angle_param_batch(params, X_batch, y_batch, layers=layers, shots=shots, batch_size=len(X_batch), qnode=qnode)
            # print_cust(f"train_epochs_angle_param_adam, params after loss computation: {params}")
            minibatch_losses.append(loss_batch)
            print_cust(f"  Mini-batch loss: {loss_batch:.4f}")

        # At the end of the epoch, compute validation loss:
        val_loss = compute_loss_angle_param_batch(params, X_val, y_val, layers=layers, shots=shots, qnode=qnode)
        validation_losses.append(val_loss)
        print_cust(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

        # Early stopping: if the validation loss doesn't improve for 'patience' epochs, stop training.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print_cust("Early stopping triggered.")
                break

    return params, minibatch_losses, validation_losses