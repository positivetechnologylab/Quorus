from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np

from quorus.training_funcs.torch_training.varquantumclassifier_training.train_step_classifier import train_step_classifier

"""### Core Training Epochs Function, Variational Classifier"""

# TODO, depthFL, 8/19, 7:04 PM: continue here

def train_models_classifier(batch_size, classifier_model, opt_classifier, criterion, train_data, train_labels, device, max_num_epochs, loss_type="depthfl"):
    """
    Trains the model for the variational quantum classifier given the training data.

    Parameters:
      batch_size: integer representing the batch size used
      classifier_model: a model object that has a method (inputs, params) and returns the output of the model
      opt_classifier: an optimizer object that can be used for optimizing classifier_model; assumed to have .backward() and .step() functions
      for computing gradients and updating parameters, respectively.
      criterion: a criterion function that takes in two inputs and computes some metric between them, used for loss.
      train_data: a 2-D tensor of shape (n_samples, dim) representing the training data.
      train_labels: a 1-D tensor of shape (n_samples) representing the training labels.
      device: a string or Device object, for where the outputs should be evaluated on.
      max_num_epochs: an integer representing the total number of epochs that should be trained.
      loss_type: a string representing the type of the loss used by this model.


    Returns:
      A dictionary containing metrics information for this training round.
    """
    # NOTE: this function mutates the inputs.
    print_cust(f"train_models_classifier, loss_type: {loss_type}")
    print_cust(f"train_models_classifier, train_data.shape: {train_data.shape}")

    print_cust(f"train_models_classifier, opt_classifier: {opt_classifier}")
    n_train_data = train_data.shape[0]

    print_cust(f"train_models_classifier, train_labels.shape: {train_labels.shape}")

    if train_data.shape[0] != train_labels.shape[0]:
      print_cust(f"train_models_classifier, amount of training data does not equal amount of training labels, train_data.shape[0]: {train_data.shape[0]}, train_labels.shape[0]: {train_labels.shape[0]}")

    print_cust(f"train_models_classifier, train_data.min(): {train_data.min()}, train_data.max(): {train_data.max()}")
    print_cust(f"train_models_classifier, train_labels.min(): {train_labels.min()}, train_labels.max(): {train_labels.max()}")
    # Collect images for plotting later

    log_metrics = {
      "disc_grad_norms": [],
      "gen_grad_norms": [],
      "disc_loss": [],
      "gen_loss": []
    }

    print_cust(f"train_models_classifier, log_metrics: {log_metrics}")

    counter = 0

    # print_cust(f"train_models, len(dataloader): {len(dataloader)}")

    print_cust(f"train_models_classifier, n_train_data: {n_train_data}")

    # max_num_epochs = math.ceil(num_iter / n_train_data)

    print_cust(f"train_models_classifier, max_num_epochs: {max_num_epochs}")

    for epoch_num in range(max_num_epochs):

        print_cust(f"train_models_classifier, [SGD] Epoch {epoch_num+1}/{max_num_epochs}")
        indices = np.arange(n_train_data)
        np.random.shuffle(indices)
        for start in range(0, n_train_data, batch_size):

            end = min(start + batch_size, n_train_data)
            batch_indices = indices[start:end]
            if len(batch_indices) < batch_size:
              print_cust(f"train_models_classifier, skipping batch because too small. len(batch_indices): {len(batch_indices)}")
              continue
            X_batch = train_data[batch_indices, :]
            y_batch = train_labels[batch_indices]
            print_cust(f"train_models_classifier, batch_indices: {batch_indices}")
            print_cust(f"train_models_classifier, X_batch.shape: {X_batch.shape}")
            print_cust(f"train_models_classifier, y_batch.shape: {y_batch.shape}")
            print_cust(f"train_models_classifier, y_batch: {y_batch}")

            # Data for training the discriminator
            # TODO: add an image processor adapter here.
            # alpha = counter / num_iter

            # FOR NOW

            # print_cust(f"data.shape: {data.shape}")
            # print_cust(f"np.linalg.norm(data[0]): {np.linalg.norm(data[0])}")
            # print_cust(f"data[0].min(): {data[0].min()}, data[0].max(): {data[0].max()}")
            # print_cust(f"data[0]: {data[0]}")
            real_data = X_batch.to(device)

            real_labels = y_batch.to(device)

            print_cust(f"train_models_classifier, X_batch.shape: {X_batch.shape}")
            print_cust(f"train_models_classifier, np.linalg.norm(real_data[0]): {np.linalg.norm(real_data[0])}")
            print_cust(f"train_models_classifier, real_data[0].min(): {real_data[0].min()}, real_data[0].max(): {real_data[0].max()}")
            print_cust(f"train_models_classifier, real_data[0]: {real_data[0]}")

            # NOTE: technically not 'semantically' correct to increment counter here, but functionally, it is the same (FOR NOW)
            print_cust(f"train_models_classifier, before counter {counter}, classifier_model: {classifier_model}")
            # TODO, layers: change this train_step call; should call train_step_classifier
            errD, disc_param_grad_norms = train_step_classifier(classifier_model, opt_classifier, criterion, real_labels, counter, real_data, loss_type=loss_type)
            print_cust(f"train_models_classifier, errD: {errD}, disc_param_grad_norms: {disc_param_grad_norms}")
            print_cust(f"train_models_classifier, after counter {counter}, classifier_model: {classifier_model}")
            counter += 1

            print_cust(f"train_models_classifier, counter: {counter}")

            log_metrics["disc_loss"].append(errD)
            log_metrics["disc_grad_norms"].append(disc_param_grad_norms)
            # if counter == num_iter:
            #     break

    print_cust(f"train_models_classifier, len(log_metrics['disc_loss']): {len(log_metrics['disc_loss'])}")
    print_cust(f"train_models_classifier, len(log_metrics['disc_grad_norms']): {len(log_metrics['disc_grad_norms'])}")
    print_cust(f"train_models_classifier, about to return, log_metrics: {log_metrics}")
    return log_metrics