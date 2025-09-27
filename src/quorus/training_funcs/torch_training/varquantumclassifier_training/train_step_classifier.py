from quorus.logging.custom_slog import print_cust
import copy
from quorus.param_processing.torch_funcs.state_dict_diffs import l2_state_dict_difference
from quorus.metrics_funcs.torch_lossfns.klloss import KLLoss
import torch

"""### Core Training Batch Function, Variational Classifier"""

def train_step_classifier(classifier_model, opt_classifier, criterion, real_labels, counter, real_data, loss_type="depthfl"):
    """
    Trains the model for one step for the variational quantum classifier given the training data.

    Parameters:
      classifier_model: a model object that has a method (inputs, params) and returns the output of the model
      opt_classifier: an optimizer object that can be used for optimizing classifier_model; assumed to have .backward() and .step() functions
      for computing gradients and updating parameters, respectively.
      criterion: a criterion function that takes in two inputs and computes some metric between them, used for loss.
      real_labels: a 1-D tensor of shape (n_samples) representing the training labels.
      counter: an integer representing the number of minibatches that have passed.
      real_data: a 2-D tensor of shape (n_samples, dim) representing the training data.
      loss_type: a string representing the type of the loss used by this model.

    Returns:
      errD, a tensor representing the loss for the model for this minibatch, and classifier_param_grad_norms, a list containing the L2 norms
      of the gradients in each layer.
    """
    # print_cust(f"train_step, disc_img_size: {disc_img_size}")
    # Training the discriminator

    print_cust(f"train_step_classifier, opt_classifier: {opt_classifier}")

    print_cust(f"train_step_classifier, loss_type: {loss_type}")

    classifier_model.zero_grad()

    # TODO: add assertions here about gradient magnitudes; should all be zero (disc at least, should)
    for name, p in classifier_model.named_parameters():
        print_cust(f"train_step_classifier, classifier_model param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step_classifier, classifier_model param name: {name}, no grad")
        else:
            print_cust(f"train_step_classifier, classifier_model param name: {name}, p.grad.norm(): {p.grad.norm()}")
            # print_cust(name, p.grad.norm())

    print_cust(f"train_step_classifier, real_data.shape: {real_data.shape}")

    classifier_beforeinf_statedict = copy.deepcopy(classifier_model.state_dict())

    # if not pca_disc:
    #     real_data = pad_images_newdim(real_data, disc_img_size ** 2)
    #     fake_data = pad_images_newdim(fake_data, disc_img_size ** 2)

    # print_cust(f"train_step, after padding, real_data.shape: {real_data.shape}")
    # print_cust(f"train_step, after padding, fake_data.shape: {fake_data.shape}")

    # TODO: detach real_data too???
    # NOTE, layers: taking that first column; the probability of observing the state to be 1.
    # DONE: TOMODIFY, DepthFL: this will be multiple probabilities; postprocess into the different losses that I want.

    outD_real = classifier_model(real_data.detach())
    if len(outD_real.shape) == 2:
      print_cust(f"train_step_classifier, outD_real.shape: {outD_real.shape}")
      outD_real = outD_real[:, 1].view(-1)
    print_cust(f"train_step_classifier, after running through classifier_model, outD_real.shape: {outD_real.shape}")
    # otherwise, outD_real is the same shape as I wanted (n_classifiers, input_dim, n_classes).


    classifier_afterinf_statedict = copy.deepcopy(classifier_model.state_dict())

    print_cust(f"train_step_classifier, real_data.shape: {real_data.shape}")
    print_cust(f"train_step_classifier, outD_real.shape: {outD_real.shape}")
    print_cust(f"train_step_classifier, outD_real: {outD_real}")
    print_cust(f"train_step_classifier, real_labels.shape: {real_labels.shape}")
    print_cust(f"train_step_classifier, real_labels: {real_labels}")

    # could do some assertions here, but can do it later
    print_cust(f"train_step_classifier, change in classifier params after inf: {l2_state_dict_difference(classifier_beforeinf_statedict, classifier_afterinf_statedict)}")

    # DONE: TOMODIFY, DepthFL: inject a new criterion. or, based on shape of the output, hardcode a loss fn for now.
    if len(outD_real.shape) == 1:
      errD_real = criterion(outD_real, real_labels)
    else:
      # TOMODIFY, DepthFL: make this on cuda() for GPU acceleration?
      criterion_kl = KLLoss()
      print_cust(f"train_step_classifier, len(outD_real.shape) != 1, outD_real.shape: {outD_real.shape}")
      errD_real = torch.zeros(1).to(classifier_model.device)
      if loss_type == "standalone":
        errD_real += criterion(outD_real[-1][:, 1], real_labels)
      elif loss_type == "depthfl":
        for one_output_idx in range(outD_real.shape[0]):
          print_cust(f"train_step_classifier, one_output_idx: {one_output_idx}")
          one_output_preds = outD_real[one_output_idx]
          print_cust(f"train_step_classifier, one_output_preds.shape: {one_output_preds.shape}")
          errD_real += criterion(one_output_preds[:, 1], real_labels)
          for second_output_idx in range(outD_real.shape[0]):
            print_cust(f"train_step_classifier, second_output_idx: {second_output_idx}")
            if second_output_idx == one_output_idx:
              continue
            second_output_preds = outD_real[second_output_idx]
            errD_real += (criterion_kl(one_output_preds, second_output_preds.detach()) / (outD_real.shape[0] - 1))

    print_cust(f"train_step_classifier, errD_real: {errD_real}")

    # NOTE, depthFL: can print out gradients at this step, to verify that they are still None.

    # Propagate gradients
    errD_real.backward()

    print_cust(f"train_step_classifier, after classifier .backward()")

    # Didn't clear grads for generator here, so that's why they r non zero I think.

    classifier_param_grad_norms = []

    for name, p in sorted(classifier_model.named_parameters(), key=lambda kv: kv[0]):
        print_cust(f"train_step_classifier, classifier_model param name: {name}, p.requires_grad: {p.requires_grad}")
        if p.grad is None:
            print_cust(f"train_step_classifier, classifier_model param name: {name}, no grad")
        else:
            print_cust(f"train_step_classifier, classifier_model param name: {name}, p.grad.norm(): {p.grad.norm()}")
            # BUG (possible), DepthFL: does p.grad.norm() require backprop, and thus consume too much memory when
            # sending back? if so, call p.grad.norm().detach().
            classifier_param_grad_norms.append(p.grad.norm())

    print_cust(f"train_step_classifier, classifier_param_grad_norms: {classifier_param_grad_norms}")

    errD = errD_real
    opt_classifier.step()
    classifier_afterdiscstep_statedict = copy.deepcopy(classifier_model.state_dict())
    print_cust(f"train_step_classifier, change in classifier params after classifier step: {l2_state_dict_difference(classifier_afterinf_statedict, classifier_afterdiscstep_statedict)}")

    # Show loss values
    print_cust(f'train_step_classifier, Iteration: {counter}, Classifier Loss: {errD.item():0.3f}')

    return errD.detach(), classifier_param_grad_norms