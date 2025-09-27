"""## QFL Experiment Parallel Function"""

import copy
import math
import pickle
from pennylane import numpy as np
import torch

from quorus.data_ops.data_loading_wrapper import load_dataset
from quorus.logging.custom_slog import print_cust
from quorus.metrics_funcs.agg_metrics_func.agg_metrics_func_eval import compute_metrics_angle_param_batch
from quorus.metrics_funcs.fid_funcs.compute_fid import compute_fid_to_data
from quorus.misc_utils.convert_data_to_lib import convert_data_to_lib
from quorus.misc_utils.create_feat_list_expansion_data import create_feat_list_expansion_data
from quorus.misc_utils.reorder_amplitude_data import reorder_amplitude_data
from quorus.qfl_utils.aggregation_funcs.federated_averaging import federated_averaging
from quorus.qfl_utils.aggregation_funcs.federated_averaging_circular_parallel_shared import federated_averaging_circular_parallel_shared
from quorus.qfl_utils.aggregation_funcs.federated_averaging_quat import federated_averaging_quat
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates import broadcast_param_updates
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared import broadcast_param_updates_shared
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared_noconv import broadcast_param_updates_shared_noconv
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared_noconv_nofinal import broadcast_param_updates_shared_noconv_nofinal
from quorus.qfl_utils.cli_params_init.initialize_client_params import initialize_client_params
from quorus.qfl_utils.data_splitting.split_data_federated import split_data_federated
from quorus.qfl_utils.federated_pca.fedpca_mocked import perform_federated_pca_mocked
from quorus.qfl_utils.meta_params_generation.generate_meta_params import generate_meta_params
from quorus.qfl_utils.meta_params_generation.generate_meta_params_random import generate_meta_params_random
from quorus.qgan_model_supp.imggen_funcs.latent_noise_gen import generate_latent_noise
import torch.nn as nn

from quorus.qgan_model_supp.imggen_funcs.save_imgs import save_tensors_to_folder
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qgan import create_qnode_qgan
from quorus.quantum_circuit_funcs.utils.qcnn_code import compute_conv_layers
from quorus.training_funcs.pennylane_training.mask_gradients import mask_gradients
from quorus.training_funcs.pennylane_training.train_mod_pennylane import train_epochs_angle_param_adam
from quorus.training_funcs.torch_training.qgan_training.qgan_training_funcs import train_models

def run_qfl_experiments_parallel(clients_config_arg, classes=["4", "9"], n_samples=1000, dataset_type="mnist", agg_strategy="fedavg", test_frac=0.2, val_frac=0.1, random_state=42, pool_in=True,
                        local_batch_size=32, local_lr=0.01, shots=1024, debug=False, init_client_data_dict=None, save_pkl=False, mask_grads=False, qubits_and_layers_to_add_block_params=[],
                        train_models_parallel=False, same_init=False, feature_skew=0.0, label_skew=None, local_pca=False, do_lda=False, feat_sel_type="top", amp_embed=False, feat_ordering="same",
                                 morepers=False, custom_debug=False, shared_pca=False, heirarchical_train=False, generative=False, use_torch=False, fed_pca_mocked=True, lr_gen=0.004, lr_disc=0.001,
                                 noise_func=generate_latent_noise, criterion_func=nn.BCELoss, targ_data_folder_prefix="testing_gen_imgs", gen_data_folder_prefix="qgan_gen_imgs", device=None, fid_batch_size=None):

  """
  Function that runs the QFL workflow for a given configuration and returns a log of the training.

  Note: in general, the client parameters dictionary is of the form:

      {
        <int>: [[conv_params, pool_params, final_pool_params,
            final_params, bias_param, block_params],
            ... for each client],
        ... for each client type
      }

  Parameters:
    clients_config_arg, a dictionary mapping client types to configuration information for clients of that type. For this code, I expect clients_config_arg to be of the form:
      {
        <int>: {
            "percentage_data": <float>,
            "num_clients": <int>,
            "local_epochs": <int>,
            "communication_rounds": <int>
        },
        ... for each client type
      }
    classes, a list of strings (preferably string representations of integers) representing the classes on which to perform classification
      ex: ["4", "9"]
    n_samples, an integer representing the number of samples to use for training
    dataset_type, a string representing the dataset type for training
      should either be "mnist", "Fashion-MNIST", "cifar10", "synthetic", "pima", "higgs", or "covertype" for those respective datasets
    agg_strategy, a string representing the aggregation scheme to perform
      should be either "fedavg", "fedavg_quat", or "fedavg_circ"
    test_frac, a float representing the fraction of the total data to be used for testing
      should be between 0 and 1
    val_frac, a float representing the fraction of each client's data to be used for validation
      should be between 0 and 1
    random_state, an integer representing the random state used in the entire program (assuming that this function generates client data -- TODO: make the use of the random
    state argument more clear)
    pool_in, a Boolean representing whether or not the model should have reduce the number of qubits to the center
    local_batch_size, an integer representing the batch size each local model should use for training
    local_lr, a float representing the learning rate each model should use for training, upon each communication round
    shots, an integer representing the number of shots each model should use (currently unused)
    debug, a Boolean indicating whether or not this function should be run in debug mode (prints the data, the amount of data per client)
    init_client_data_dict, a dictionary that maps client types to lists of data for each client that is used for each client
      should be of the form
      {
        <int>: [[(X_train, y_train), (X_val, y_val), (pca_obj, pca_reduced_data)],
                ... for each client
              ],
        ... for each client type
      }
    save_pkl, a Boolean indicating whether or not the data should be saved to a pickle file
    mask_grads, a Boolean indicating whether or not the gradients should have a Boolean mask (and parameters in smaller qubit sets should not be updated)
    qubits_and_layers_to_add_block_params, a dictionary that maps client types to a list of (n_qubits, n_layers) representing the block parameters that each client has
    train_models_parallel, a Boolean indicating whether or not the models should be trained in parallel (for personalized models)
    same_init, a Boolean indicating whether or not all the clients should start with the same initial parameters
    feature_skew, a float between 0 and 1 specifying the magnitude of the feature skew (i.e., the strength of sorting the features by the first feature and how prominently that appears
    in the data for clients)
    label_skew, a float between 0 and 1 specifying the magnitude of the label skew that each client has
    do_lda, a Boolean indicating whether or not to perform random sketching
    feat_sel_type, a string representing the choice of features to make in angle encoding for each client
      should be either "top" for selecting the features with the highest variance, or "toplow" for selecting half the features with the highest variance and half the features with the
      lowest variance at each expansion
    amp_embed, a Boolean indicating whether the data should be amplitude encoded
    feat_ordering, a string representing whether the features in amplitude encoding should be taken as-is, or if it should be taken in a different order
      should be either "same" for as-is feature ordering, or "highest_var" for sorting the features in descending order in terms of highest variance
    morepers, a Boolean indicating whether or not only the the convolutional parameters should or should not be aggregated (if not, the convolutional parameters are unique to each client,
    so each client has its own personalized convolutional parameters -- see PDF drawing sent in Slack)

    Returns:
      data_logs, a dictionary of the following form:
        {
          'clients_data_dict': <client_data_dict> of the above form,
          'testing_data': (X_test, y_test),
          0:
            {
              "aggregated_params": <params>,
              <int>: {
                "local_epochs": <int>,
                "client_metrics": [
                    {
                      "trained_params": <params>,
                      "minibatch_losses": list<float>,
                      "validation_losses": list<float>,
                      "training_acc": <float>,
                      "testing_acc": <float>,
                      "testing_loss": <float>,
                      "training_acc_stdev": <float>,
                      "testing_acc_stdev": <float>,
                      "training_acc_topk": <float>,
                      "testing_acc_topk": <float>
                    }, ... (for each client)
                ]
              },
              ... for each client type
            },
          ... for each communication round
          clients_config_arg["communication_rounds"] - 1: (same format as above data logs dictionary)
        }
  """

  # TOADD, generative:
  # 1. An option to this function to indicate that we are having a generative model

  print_cust(f"run_qfl_experiments_parallel, heirarchical_train: {heirarchical_train}")
  # Find the maximum, minimum size clients, as well as the total communication rounds
  max_size_clients = max(clients_config_arg.keys())
  min_size_clients = min(clients_config_arg.keys())

  if custom_debug:
    print_cust(f"run_qfl_experiments_parallel, max_size_clients: {max_size_clients}, min_size_clients, {min_size_clients}")
    assert max_size_clients >= min_size_clients, "Maximum sized client is NOT at least as large as minimum sized client"

  num_total_rounds = max([clients_config_arg[key]["communication_rounds"] for key in clients_config_arg])

  # Note: this does impose some more constraints on the format in which the round info is passed (larger MUST be > smaller; training 4 implies training 8.) going to stick with this for now
  total_rounds_accum = 0
  client_types_to_rounds = {}
  for client_type in sorted(clients_config_arg.keys()):
    client_rounds = clients_config_arg[client_type]["communication_rounds"]
    client_types_to_rounds[client_type] = client_rounds - total_rounds_accum
    total_rounds_accum += client_rounds

  # this should be client type to rounds TO RUN for basically clients that size AND ABOVE.
  print_cust(f"run_qfl_experiments_parallel, client_types_to_rounds: {client_types_to_rounds}")

  if custom_debug:
    print_cust(f"run_qfl_experiments_parallel, num_total_rounds: {num_total_rounds}")
    assert num_total_rounds >= 0, "Number of total rounds is not at least 0"

  # The number of output qubits is enough to accommodate the number of classes.
  n_output_qubits = int(np.ceil(np.log2(len(classes))))

  if custom_debug:
    print_cust(f"run_qfl_experiments_parallel, n_output_qubits: {n_output_qubits}")
    assert n_output_qubits >= 0, "Number of output qubits is not at least 0"

  print_cust(f"run_qfl_experiments_parallel, num_total_rounds: {num_total_rounds}")

  print_cust(f"run_qfl_experiments_parallel, n_output_qubits: {n_output_qubits}")

  # Load the input dataset. If we are doing amplitude embedding or local PCA, then we do not want to dimensionality reduce the input images.
  # Note that the pixels are normalized to be between 0 and 1.
  keep_orig_imgs = (local_pca or amp_embed)

  if custom_debug:
    print_cust(f"run_qfl_experiments_parallel, keep_orig_imgs: {keep_orig_imgs}")

  X_angles, y = load_dataset(dataset_type=dataset_type, classes=classes, n_samples=n_samples, num_feats=max_size_clients, keep_orig_imgs=keep_orig_imgs, custom_debug=custom_debug)

  print_cust(f"run_qfl_experiments_parallel, X_angles.shape: {X_angles.shape}")

  print_cust(f"run_qfl_experiments_parallel, local_pca: {local_pca}, do_lda: {do_lda}")

  # Load in the previous data dictionary, if supplied.
  if init_client_data_dict is not None:
    clients_data_dict = init_client_data_dict["clients_data_dict"]
    (X_test, y_test) = init_client_data_dict["testing_data"]
    # if shared_pca:
    #   shared_max_comps = init_client_data_dict['shared_max_comps']
    #   shared_min_comps = init_client_data_dict['shared_min_comps']
    print_cust(f"run_qfl_experiments_parallel, loaded in existing data")
  else:
    clients_data_dict, (X_test, y_test) = split_data_federated(X_angles, y, clients_config_arg, test_frac, val_frac=val_frac, random_state=random_state, feature_skew=feature_skew, label_skew=label_skew, local_pca=local_pca,
                                                              do_lda=do_lda, feat_sel_type=feat_sel_type, amp_embed=amp_embed, feat_ordering=feat_ordering, shared_pca=shared_pca, fed_pca_mocked=fed_pca_mocked)
    print_cust(f"run_qfl_experiments_parallel, generated new data")

  # Create a log of the information that we have throughout training.
  data_logs = {}

  data_logs["clients_data_dict"] = clients_data_dict
  data_logs["testing_data"] = (X_test, y_test)
  # if shared_pca:
  #   data_logs['shared_max_comps'] = shared_max_comps
  #   data_logs['shared_min_comps'] = shared_min_comps

  # Store the initial set of training parameters and data in a pickle file.
  if save_pkl:
    with open(f"data_logs_n_samples_{n_samples}_dataset_type_{dataset_type}_classes_{'_'.join(classes)}_train_models_parallel_{train_models_parallel}_feature_skew_{feature_skew}_label_skew_{label_skew}_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_{generative}.pkl", "wb") as file:
      pickle.dump(data_logs, file)


  if use_torch:
    math_int = torch
  else:
    math_int = np

  # Maybe right here, have a torch data conversion function ??? (if not done in perform_federated_pca_mocked)
  clients_data_dict = convert_data_to_lib(clients_data_dict, math_int=math_int)
  if math_int == torch:
    X_test = torch.tensor(X_test, dtype=torch.float32)

  if generative:
    print_cust(f"run_qfl_experiments_parallel, X_test.shape: {X_test.shape}")
    if len(X_test.shape) == 2:
      img_dim = math.isqrt(X_test.shape[1])
      assert (img_dim ** 2) == X_test.shape[1], f"run_qfl_experiments_parallel, X_test is not a perfect square, X_test.shape: {X_test.shape}"
      X_test = X_test.view(X_test.shape[0], img_dim, img_dim)
    print_cust(f"run_qfl_experiments_parallel, generative, X_test.max(): {X_test.max()}, X_test.min(): {X_test.min()}")
    save_tensors_to_folder(X_test, f"{targ_data_folder_prefix}", "img")
    n_samples_test = X_test.shape[0]


  # TODO: continue reading here, 7/9
  # also, please put this in a function......
  if shared_pca:
    # NOTE: this function MUTATES clients_data_dict
    pca_global, shared_min_comps, shared_max_comps, inv_pca_min, inv_pca_max = perform_federated_pca_mocked(clients_data_dict, max_size_clients, random_seed=random_state, math_int=math_int, device=device)
    # TODO: ... = perform_federated_pca_mocked()...

  # For sanity, print out the data as well as the amount of data that each client has.
  if debug:
    for client_type in clients_data_dict:
      clients_data_list = clients_data_dict[client_type]
      print_cust(f"client_type: {client_type}, len(clients_data_list): {len(clients_data_list)}")
      for client_idx in range(len(clients_data_list)):
        client_data = clients_data_list[client_idx]
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, len(client_data): {len(client_data)}")
        if not local_pca:
          (training_data, validation_data) = client_data
        else:
          (training_data, validation_data, pca_info) = client_data
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, len(training_data): {len(training_data)}, len(validation_data): {len(validation_data)}")
        (X_train, y_train) = training_data
        (X_val, y_val) = validation_data
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")

  # Initialize the parameters for each client.
  # qnode_func, device, pca_info
  qnode_func = None
  # device = None
  pca_info = ()
  if generative:
    qnode_func = create_qnode_qgan
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pca_info = (pca_global, shared_min_comps, shared_max_comps, inv_pca_min, inv_pca_max)
  client_params_dict = initialize_client_params(clients_config_arg, model_size=min_size_clients, cur_client_params_dict=None, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params, train_models_parallel=train_models_parallel,
                                                n_output_qubits=n_output_qubits, generative=generative, use_torch=use_torch, qnode_func=qnode_func, device=device, pca_info=pca_info)
  print_cust(f"run_qfl_experiments_parallel, first initialization of client params, client_params_dict: {client_params_dict}")

  # Make all parameters the same across clients with a new random initialization if we want clients to have the same initialization.
  if same_init:
    print_cust(f"run_qfl_experiments_parallel, same parameters initialization")
    with torch.no_grad():
      meta_params = generate_meta_params(client_params_dict, clients_data_dict, math_int=math_int)
      meta_params = generate_meta_params_random(meta_params, math_int=math_int)
      client_params_dict = broadcast_param_updates(client_params_dict, meta_params)
      print_cust(f"run_qfl_experiments_parallel, client_params_dict: {client_params_dict}")

  # client_types = sorted(list(clients_config_arg.keys()))

  # Initialize the data logging information for each communication round.

  for round_num in range(num_total_rounds):
    print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}")
    # Initialize the data logging information for this communication round.
    data_logs[round_num] = {}
    data_logs[round_num]["aggregated_params"] = None
    for client_size, cfg in clients_config_arg.items():
      data_logs[round_num][client_size] = {}
      # TODO: add a field for testing loss in data_logs
      data_logs[round_num][client_size]["local_epochs"] = cfg["local_epochs"]
      data_logs[round_num][client_size]["client_metrics"] = []
      for client_idx in range(cfg["num_clients"]):
        data_logs[round_num][client_size]["client_metrics"].append({"trained_params": None,
                                                      "minibatch_losses": None,
                                                      "validation_losses": None,
                                                      "training_acc": None,
                                                      "testing_acc": None,
                                                      "testing_loss": None,
                                                      "training_acc_stdev": None,
                                                      "testing_acc_stdev": None,
                                                      "training_acc_topk": None,
                                                      "testing_acc_topk": None,
                                                      "testing_fid": None})

  rounds_elapsed = 0
  for cur_shared_model_size, model_size_rounds in client_types_to_rounds.items():

    if cur_shared_model_size > min_size_clients and not train_models_parallel:
      # two things to check:
      # 1. initialization with existing parameters - OK
      # 2. expansion from smaller to larger qubits. - easy fix in pca_rescaler
      client_params_dict = initialize_client_params(clients_config_arg, model_size=cur_shared_model_size, cur_client_params_dict=client_params_dict, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params, train_models_parallel=train_models_parallel,
                                                n_output_qubits=n_output_qubits, generative=generative, use_torch=use_torch, qnode_func=qnode_func, device=device, pca_info=pca_info)
      print_cust(f"run_qfl_experiments_parallel, after expansion, client_params_dict: {client_params_dict}")
      existing_clients_datadict = list(clients_data_dict.keys())
      for existing_client_size in existing_clients_datadict:
        if cur_shared_model_size > existing_client_size:
          del clients_data_dict[existing_client_size]

      print_cust(f"run_qfl_experiments_parallel, not train_models_parallel, clients_data_dict.keys(): {clients_data_dict.keys()}")

    # For each communication round, iterate over each client and have each client train on their dataset for the specified number of epochs.
    for round_num in range(model_size_rounds):
      round_num += rounds_elapsed
      print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}")

      for client_type, client_params in client_params_dict.items():
        # TODO: add a condition, that, based on the current model size, if heirarchical, skips particular client types.
        print_cust(f"run_qfl_experiments_parallel, client_type: {client_type}")
        # Get the clients' data and the number of convolutional layers for clients of this type.
        client_data = clients_data_dict[client_type]
        if train_models_parallel:
          cur_model_size = client_type
        else:
          cur_model_size = cur_shared_model_size
        print_cust(f"run_qfl_experiments_parallel, cur_model_size: {cur_model_size}")
        conv_layers = compute_conv_layers(cur_model_size, n_output_qubits, generative=generative)
        print_cust(f"run_qfl_experiments_parallel, conv_layers: {conv_layers}")
        for client_idx in range(len(client_params)):
          print_cust(f"run_qfl_experiments_parallel, client_idx: {client_idx}")
          # Get the training data and parameters for each client.
          client_params_indiv = client_params[client_idx]
          client_data_indiv = client_data[client_idx]

          client_train_data = client_data_indiv[0]
          client_val_data = client_data_indiv[1]

          if debug:
            print_cust(f"run_qfl_experiments_parallel, len(client_train_data): {len(client_train_data)}")
            print_cust(f"run_qfl_experiments_parallel, client_train_data: {client_train_data}")
            print_cust(f"run_qfl_experiments_parallel, len(client_val_data): {len(client_val_data)}")
            print_cust(f"run_qfl_experiments_parallel, client_val_data: {client_val_data}")

          grad_mask = None

          # Gradient masking hasn't been tested for a while, so currently is commented out.
          # TODO: implement gradient masking!!!!!!

          if mask_grads:
            if cur_model_size > min_size_clients and not generative:
              grad_mask = mask_gradients(client_params_indiv)

          print_cust(f"run_qfl_experiments_parallel, grad_mask: {grad_mask}")

          # Create the feature list indicating what features for which qubits are to be used for this client, as well as the expansion data (used for identity
          # initialization).
          expand = False
          if cur_model_size > min_size_clients:
            expand = True
          if not generative:
            feature_list, expansion_data = create_feat_list_expansion_data(cur_model_size, conv_layers, expand=expand, pool_in=pool_in, min_qubits_noexpand=min_size_clients,
                                                                          train_models_parallel=train_models_parallel, feat_sel_type=feat_sel_type)
          else:
            feature_list, expansion_data = list(range(cur_model_size)), []

          # Create the QNode for this client.
          # NOTE: for generative, there is no need to create a separate qnode; it should already be created for the generative model, upon instantiation.
          if not generative:
            qnode = create_qnode_qcnn(cur_model_size, conv_layers, expansion_data, n_classes=len(classes))

          num_local_epochs = clients_config_arg[client_type]["local_epochs"]

          if not generative:
            client_params_indiv = copy.deepcopy(client_params_indiv)

          # If amplitude embedding the data, reorder the data; otherwise, sample features based on feature_list.
          if amp_embed:
            training_data = reorder_amplitude_data(client_train_data[0], feature_list)
            validation_data = reorder_amplitude_data(client_val_data[0], feature_list)
          else:
            training_data = client_train_data[0][:, feature_list]
            validation_data = client_val_data[0][:, feature_list]

          # Train the client for the specified number of epochs.
          # TODO: update the grad_mask here.
          if not generative:
            trained_params, minibatch_losses, validation_losses = train_epochs_angle_param_adam(
          client_params_indiv, training_data, client_train_data[1], validation_data, client_val_data[1],
          n_epochs=num_local_epochs, shots=shots, batch_size=local_batch_size, lr=local_lr, qnode=qnode, trainable_mask=grad_mask)
            # Update the client's trained parameters.
            client_params[client_idx] = trained_params
          else:
            # I mean, I could return the minibatch_losses, validation_losses, but idt I will.
            print_cust(f"run_qfl_experiments_parallel, generative is True (QGAN training)")
            client_generator = client_params_indiv[5][0]
            print_cust(f"run_qfl_experiments_parallel, type(client_generator): {type(client_generator)}")
            client_discriminator = client_params_indiv[5][1]
            print_cust(f"run_qfl_experiments_parallel, type(client_discriminator): {type(client_discriminator)}")
            print_cust(f"run_qfl_experiments_parallel, nn.ParameterList([client_generator.q_params[-1]]: {nn.ParameterList([client_generator.q_params[-1]])}")
            client_optim_gen = torch.optim.SGD(nn.ParameterList([client_generator.q_params[-1]]), lr=lr_gen)
            client_optim_disc = torch.optim.SGD(client_discriminator.parameters(), lr=lr_disc)
            # maybe concatenate in the train with val data? not using val data otherwise
            # can use test_result_imgs for visualization, later, if I'd like.
            test_result_imgs = train_models(client_generator.n_qubits_gen, local_batch_size, client_generator, client_discriminator, client_optim_disc, client_optim_gen,
                                            noise_func=noise_func, criterion=criterion_func(), train_data=training_data, device=device, image_size=0, compressed_img_size=0, max_num_epochs=num_local_epochs,
                                            n_qubits_small=0, gen_pcas=True, disc_img_size=None, pca_disc=True)

          # Perform PCA on the testing data, if specified.
          if local_pca and not generative:
            client_pca_info = client_data_indiv[2]
            client_pca = client_pca_info[0]
            client_data_pca = client_pca_info[1]
            X_test_client_pca = client_pca.transform(X_test)
            # Scale each PCA component independently to [0, π]
            X_test_client_angle = math_int.zeros_like(X_test_client_pca)
            # assuming that the number of components is simply the client's data type
            # TODO: rescale the testing data to match the PCA scale for each client
            for i in range(client_type):
                comp = X_test_client_pca[:, i]
                if shared_pca:
                  lo, hi = shared_min_comps[i], shared_max_comps[i]
                  comp_norm = ( (comp - lo) / (hi - lo + 1e-8) )
                  # comp_norm = np.clip(comp_norm, 0, 1)
                else:
                  orig_comp = client_data_pca[:, i]
                  comp_norm = (comp - orig_comp.min()) / (orig_comp.max() - orig_comp.min() + 1e-8)
                X_test_client_angle[:, i] = comp_norm * math_int.pi
          else:
            X_test_client_angle = X_test

          # Select the features for amplitude encoding based on feat_ordering.
          if amp_embed:
            if feat_ordering == "highest_var":
              variances = X_angles.var(axis=0, ddof=0)
              order = np.argsort(variances)[::-1]
              X_test_client_angle = X_test_client_angle[:, order] + 1e-3
            testing_data = reorder_amplitude_data(X_test_client_angle, feature_list)
          else:
            if not generative:
              testing_data = X_test_client_angle[:, feature_list]

            # TODO: encapsulate this logic in a helper function


          # train_acc = compute_avg_acc_angle_param_batch(trained_params, client_train_data[0][:, feature_list], client_train_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # test_acc = compute_avg_acc_angle_param_batch(trained_params, X_test_client_angle[:, feature_list], y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # test_loss = compute_loss_angle_param_batch(trained_params, X_test_client_angle[:, feature_list], y_test, shots=shots, qnode=qnode)

          # train_acc_stdev = compute_std_acc_angle_param_batch(trained_params, client_train_data[0][:, feature_list], client_train_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # test_acc_stdev = compute_std_acc_angle_param_batch(trained_params, X_test_client_angle[:, feature_list], y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # train_acc_topk = compute_top_k_acc_angle_param_batch(trained_params, client_train_data[0][:, feature_list], client_train_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # test_acc_topk = compute_top_k_acc_angle_param_batch(trained_params, X_test_client_angle[:, feature_list], y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

          # Compute the training and testing metrics.
          if not generative:
            train_acc, train_acc_stdev, train_acc_topk, train_loss = compute_metrics_angle_param_batch(trained_params, training_data, client_train_data[1], layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)

            # Store the training and testing metrics.
            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["training_acc"] = train_acc
            # data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["testing_acc"] = test_acc
            # data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["testing_loss"] = test_loss

            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["training_acc_stdev"] = train_acc_stdev
            # data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["testing_acc_stdev"] = test_acc_stdev

            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["training_acc_topk"] = train_acc_topk
            # data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["testing_acc_topk"] = test_acc_topk

            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["trained_params"] = trained_params
            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["minibatch_losses"] = minibatch_losses
            data_logs[round_num][cur_model_size]["client_metrics"][client_idx]["validation_losses"] = validation_losses

      # perform aggregation
      # NOTE: assume, during aggregation, that the parameters for all the clients have the same shape.
      # Aggregate parameters based on the strategy provided.
      print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, parameter aggregation, agg_strategy: {agg_strategy}")

      print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, before aggregation, client_params_dict: {client_params_dict}")

      if agg_strategy == "fedavg":
        aggregated_params = federated_averaging(client_params_dict, clients_data_dict)
      elif agg_strategy == "fedavg_quat":
        aggregated_params = federated_averaging_quat(client_params_dict, clients_data_dict)
      elif agg_strategy == "fedavg_circ":
        aggregated_params = federated_averaging_circular_parallel_shared(client_params_dict, clients_data_dict, generative=generative)
        # print_cust(f"run_qfl_experiments, aggregated_params: {aggregated_params}")

      print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, after aggregation, aggregated_params: {aggregated_params}")

      # # broadcast parameter updates
      # for client_type, client_params in client_params_dict.items():
      #   for client_idx in range(len(client_params)):
      #     client_params[client_idx] = aggregated_params

      # client_params_dict = broadcast_param_updates_shared(client_params_dict, aggregated_params)

      # Broadcast parameters based on the personalization scheme.
      if morepers == "aggshared":
        client_params_dict = broadcast_param_updates_shared(client_params_dict, aggregated_params)
      elif morepers == "aggnoconv":
        client_params_dict = broadcast_param_updates_shared_noconv(client_params_dict, aggregated_params)
      elif morepers == "aggnoconvnofinal":
        client_params_dict = broadcast_param_updates_shared_noconv_nofinal(client_params_dict, aggregated_params)

      print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, after broadcasting, client_params_dict: {client_params_dict}")

      # Store the aggregated parameters for this round.
      # TODO: aggregated parameters don't have a significant meaning as not all of them are broadcasted to clients based on the broadcasting/personalization scheme
      data_logs[round_num]["aggregated_params"] = aggregated_params


      # TODO: recompute metrics of the SHARED model for EACH client.
      one_cli_size = list(client_params_dict.keys())[0]

      if generative:
        # NOTE: this assumes that all generators and discriminators have the same parameters.
        agg_model = client_params_dict[one_cli_size][0][5][0]
        # agg_disc = client_params_dict[one_cli_size][0][5][1]
      else:
        agg_model = client_params_dict[one_cli_size][0]


      test_acc, test_acc_stdev, test_acc_topk, test_loss, test_fid_score = (None, None, None, None, None)

      if fid_batch_size is None: # for now, for debug
        fid_batch_size = n_samples_test

      if num_total_rounds > 10:
        if (round_num + 1) % 10 == 0:
          if not generative:
            # TODO: this is broken for the non-generative case. process the testing data outside of the loop for each client.. not doing that now tho; not too relevant for
            # generative case
            test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(agg_model, testing_data, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)
          else:
            folder_id_suffix = f"round_num_{round_num}_cur_shared_model_size_{cur_shared_model_size}"
            test_fid_score = compute_fid_to_data(agg_model, noise_func, f"{targ_data_folder_prefix}",
                                                f"{gen_data_folder_prefix}/{folder_id_suffix}", n_samples_test, device, fid_batch_size=fid_batch_size)
        else:
          test_acc, test_acc_stdev, test_acc_topk, test_loss = (None, None, None, None)
      else:
        if not generative:
          test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(agg_model, testing_data, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode)
        else:
          folder_id_suffix = f"round_num_{round_num}_cur_shared_model_size_{cur_shared_model_size}"
          test_fid_score = compute_fid_to_data(agg_model, noise_func, f"{targ_data_folder_prefix}",
                                              f"{gen_data_folder_prefix}/{folder_id_suffix}", n_samples_test, device, fid_batch_size=fid_batch_size)
      if not generative:
        print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, test_acc: {test_acc}, test_loss: {test_loss}, test_acc_stdev: {test_acc_stdev}, test_acc_topk: {test_acc_topk}")
      else:
        print_cust(f"run_qfl_experiments_parallel, round_num: {round_num}, cur_model_size: {cur_model_size}, test_fid_score: {test_fid_score}")

    rounds_elapsed += model_size_rounds

  return data_logs