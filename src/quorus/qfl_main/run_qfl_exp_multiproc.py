"""## QFL Experiment Parallel Multiprocess"""

"""### QFL Experiment Parallel Multiprocess Function"""

from concurrent.futures import as_completed
import copy
import math
import os
import pickle

from pennylane import numpy as np
import torch
import torch.nn as nn

from quorus.data_ops.data_loading_wrapper import load_dataset
from quorus.logging.custom_slog import print_cust
from quorus.metrics_funcs.agg_metrics_func.agg_metrics_func_eval import compute_metrics_angle_param_batch
from quorus.metrics_funcs.fid_funcs.compute_fid import compute_fid_to_data
from quorus.misc_utils.convert_data_to_lib import convert_data_to_lib
from quorus.qfl_main.utils.dummyexecutor import DummyExecutor
from quorus.qfl_main.utils.train_single_client import _train_single_client
from quorus.qfl_utils.aggregation_funcs.federated_averaging import federated_averaging
from quorus.qfl_utils.aggregation_funcs.federated_averaging_circular_parallel_shared import federated_averaging_circular_parallel_shared
from quorus.qfl_utils.aggregation_funcs.federated_averaging_quat import federated_averaging_quat
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates import broadcast_param_updates
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared import broadcast_param_updates_shared
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared_noconv import broadcast_param_updates_shared_noconv
from quorus.qfl_utils.broadcast_params_funcs.broadcast_param_updates_shared_noconv_nofinal import broadcast_param_updates_shared_noconv_nofinal
from quorus.qfl_utils.cli_optimizers_init.initialize_client_optimizers import initialize_client_optimizers
from quorus.qfl_utils.cli_params_init.initialize_client_params import initialize_client_params
from quorus.qfl_utils.data_splitting.split_data_federated import split_data_federated
from quorus.qfl_utils.federated_pca.fedpca_mocked import perform_federated_pca_mocked
from quorus.qfl_utils.meta_params_generation.generate_meta_params import generate_meta_params
from quorus.qfl_utils.meta_params_generation.generate_meta_params_random import generate_meta_params_random
from quorus.qgan_model_supp.imggen_funcs.latent_noise_gen import generate_latent_noise
from quorus.qgan_model_supp.imggen_funcs.save_imgs import save_tensors_to_folder
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_cheating import create_qnode_qcnn_multieval_cheating
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_multirun import create_qnode_qcnn_multieval
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qgan import create_qnode_qgan
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_singlerun_multiprobs import create_qnode_qcnn_singleeval
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_singlerun_tunnel import create_qnode_qcnn_singleeval_tunneldown
from quorus.quantum_circuit_funcs.utils.qcnn_code import compute_conv_layers
from quorus.training_funcs.pennylane_training.mask_gradients import mask_gradients

def run_qfl_experiments_parallel_multiprocess(clients_config_arg, classes=["4", "9"], n_samples=1000, dataset_type="mnist", agg_strategy="fedavg", test_frac=0.2, val_frac=0.1, random_state=42, pool_in=True,
                        local_batch_size=32, local_lr=0.01, shots=1024, debug=False, init_client_data_dict=None, save_pkl=False, mask_grads=False, qubits_and_layers_to_add_block_params=[],
                        train_models_parallel=False, same_init=False, feature_skew=0.0, label_skew=None, local_pca=False, do_lda=False, feat_sel_type="top", amp_embed=False, feat_ordering="same",
                                 morepers=False, custom_debug=False, shared_pca=False, heirarchical_train=False, generative=False, use_torch=False, fed_pca_mocked=True, lr_gen=0.004, lr_disc=0.001,
                                 noise_func=generate_latent_noise, criterion_func=nn.BCELoss, targ_data_folder_prefix="testing_gen_imgs", gen_data_folder_prefix="qgan_gen_imgs", device=None, fid_batch_size=None,
                                              max_workers=None, mp_ctx=None, log_data_folder="local_placeholder", initial_supp_params=None, optim_type="sgd", gen_betas=(0.5, 0.9), disc_betas=(0.5, 0.9),
                                              resc_invpca=True, compute_fid=True, is_qcnn=True, pennylane_interface="autograd", opt_layers=[-1], alt_zeros_init="", multiclassifier_type="", testacc_rd_cutoff=101,
                                              qubits_and_layer_types_block_params=[], loss_type="depthfl", lr_disc_decay=1.0, cont_optim_state=False):

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
      so each client has its own personalized convolutional parameters)
      custom_debug, a Boolean indicating whether or not additional debug information should be turned on
      shared_pca, a Boolean indicating whether or not shared PCA should be performed among clients
      heirarchical_train, a Boolean indicating whether or not clients should be trained in order of their types
      generative, a Boolean indicating whether or not a generative model should be used
      use_torch, a Boolean indicating whether or not PyTorch should be used throughout the code
      fed_pca_mocked, a Boolean indicating whether or not a mocked implementation of Federated PCA should be used
      lr_gen, a float specifiying the learning rate of the generator for the QGAN setup
      lr_disc, a float specifying the learning rate of the discriminator for the QGAN setup (or of the classifier for the Variational Quantum Classifier setup)
      noise_func, a function specifying the latent noise generation function to be used
      criterion_func, a function specifying the loss criterion for the classifier
      targ_data_folder, a string specifying the folder where testing generated images are saved, in the QGAN setup
      gen_data_folder_prefix, a folder specifying the prefix for where the generated images from the QGAN are saved
      device, a string or Device object specifying the device used
      fid_batch_size, an integer or None specifying the batch size used for FID calculation
      max_workers, an integer or None specifying the maximum number of workers used 
      mp_ctx, a Context object specifying the context used for multiprocessing
      log_data_folder, a string representing the folder path for which data is to be stored
      initial_supp_params, a dictionary specifying the initial parameters used (or None, if not provided)
      optim_type, a string specifying the type of optimizer to be used
      gen_betas, a tuple of floats specifying the betas used for the generator optimizer
      disc_betas, a tuple of floats specifying the betas used for the discrimiantor optimizer
      resc_invpca, a Boolean indicating whether or not generated images should be rescaled based on the PCA object, for the QGAN setup
      compute_fid, a Boolean indicating whether or not FID should be computed
      is_qcnn, a Boolean indicating whether or not the QCNN model should be used.
      pennylane_interface, a string specifying the interface that Pennylane should use.
      opt_layers, a List[int] or None specifying the layers that each client should optimize over.
      alt_zeros_init, a string specifying the initialization options for the client parameters. This is an experimental feature.
      multiclassifier_type, a string specifying the type of qnode that should be run when extracting multiple classifier outputs
      from the quantum circuit.
      testacc_rd_cutoff, an integer specifying the threshold for which testing accuracy is to be computed; beyond the threshold,
      testing accuracy is computed every 10 rounds.
      qubits_and_layer_types_block_params, a dictionary mapping each client type to a list of strings representing the type of the variational
      layers it uses.
      loss_type, a string representing the type of loss to be used, for each client classifier (in the variational classifier case).
      lr_disc_decay, an integer specifying the decay rate of the discriminator learning rate, applied every round.
      cont_optim_state, a Boolean specifying whether the optimizer state should persist across rounds.

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
    print_cust(f"run_qfl_experiments_parallel_multiprocess, alt_zeros_init: {alt_zeros_init}")
    print_cust(f"run_qfl_experiments_parallel_multiprocess, opt_layers: {opt_layers}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, compute_fid: {compute_fid}")
    print_cust(f"run_qfl_experiments_parallel_multiprocess, lr_gen: {lr_gen}, lr_disc: {lr_disc}")
    print_cust(f"run_qfl_experiments_parallel_multiprocess, resc_invpca: {resc_invpca}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, lr_disc_decay: {lr_disc_decay}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, cont_optim_state: {cont_optim_state}")

    # TOADD, generative:
    # 1. An option to this function to indicate that we are having a generative model

    # TOADD, layers: an argument for the classification circuit WITHOUT QCNN part.

    # TOMODIFY, layers (NOT strictly necessary, for now): add an argument to this function specifying we are doing layer expansion

    print_cust(f"run_qfl_experiments_parallel_multiprocess, generative: {generative}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, mp_ctx: {mp_ctx}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, heirarchical_train: {heirarchical_train}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, log_data_folder: {log_data_folder}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, optim_type: {optim_type}")
    # Find the maximum, minimum size clients, as well as the total communication rounds
    max_size_clients = max(clients_config_arg.keys())
    min_size_clients = min(clients_config_arg.keys())

    if custom_debug:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, max_size_clients: {max_size_clients}, min_size_clients, {min_size_clients}")
      assert max_size_clients >= min_size_clients, "Maximum sized client is NOT at least as large as minimum sized client"

    num_total_rounds = max([clients_config_arg[key]["communication_rounds"] for key in clients_config_arg])

    # Note: this does impose some more constraints on the format in which the round info is passed (larger MUST be > smaller; training 4 implies training 8.) going to stick with this for now
    total_rounds_accum = 0
    client_types_to_rounds = {}
    for client_type in sorted(clients_config_arg.keys()):
      client_rounds = clients_config_arg[client_type]["communication_rounds"]
      client_types_to_rounds[client_type] = client_rounds - total_rounds_accum
      total_rounds_accum += (client_rounds - total_rounds_accum)

    # this should be client type to rounds TO RUN for basically clients that size AND ABOVE.
    print_cust(f"run_qfl_experiments_parallel_multiprocess, client_types_to_rounds: {client_types_to_rounds}")

    if custom_debug:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, num_total_rounds: {num_total_rounds}")
      assert num_total_rounds >= 0, "Number of total rounds is not at least 0"

    # The number of output qubits is enough to accommodate the number of classes.
    n_output_qubits = int(np.ceil(np.log2(len(classes))))

    if custom_debug:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, n_output_qubits: {n_output_qubits}")
      assert n_output_qubits >= 0, "Number of output qubits is not at least 0"

    print_cust(f"run_qfl_experiments_parallel_multiprocess, num_total_rounds: {num_total_rounds}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, n_output_qubits: {n_output_qubits}")

    # Load the input dataset. If we are doing amplitude embedding or local PCA, then we do not want to dimensionality reduce the input images.
    # Note that the pixels are normalized to be between 0 and 1.
    keep_orig_imgs = (local_pca or amp_embed)

    if custom_debug:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, keep_orig_imgs: {keep_orig_imgs}")

    X_angles, y = load_dataset(dataset_type=dataset_type, classes=classes, n_samples=n_samples, num_feats=max_size_clients, keep_orig_imgs=keep_orig_imgs, custom_debug=custom_debug)

    print_cust(f"run_qfl_experiments_parallel_multiprocess, X_angles.shape: {X_angles.shape}")

    print_cust(f"run_qfl_experiments_parallel_multiprocess, local_pca: {local_pca}, do_lda: {do_lda}")

    # Load in the previous data dictionary, if supplied.
    if init_client_data_dict is not None:
      clients_data_dict = init_client_data_dict["clients_data_dict"]
      (X_test, y_test) = init_client_data_dict["testing_data"]
      # if shared_pca:
      #   shared_max_comps = init_client_data_dict['shared_max_comps']
      #   shared_min_comps = init_client_data_dict['shared_min_comps']
      print_cust(f"run_qfl_experiments_parallel_multiprocess, loaded in existing data")
    else:
      filtered_clients_config_arg = {}
      for cli_type, cli_config_val in clients_config_arg.items():
        if cli_config_val["percentage_data"] > 0.0:
          filtered_clients_config_arg[cli_type] = cli_config_val
      print_cust(f"run_qfl_experiments_parallel_multiprocess, filtered_clients_config_arg: {filtered_clients_config_arg}")
      clients_data_dict, (X_test, y_test) = split_data_federated(X_angles, y, filtered_clients_config_arg, test_frac, val_frac=val_frac, random_state=random_state, feature_skew=feature_skew, label_skew=label_skew, local_pca=local_pca,
                                                                do_lda=do_lda, feat_sel_type=feat_sel_type, amp_embed=amp_embed, feat_ordering=feat_ordering, shared_pca=shared_pca, fed_pca_mocked=fed_pca_mocked)
      print_cust(f"run_qfl_experiments_parallel_multiprocess, generated new data")

    # Create a log of the information that we have throughout training.
    data_logs = {}

    data_logs["clients_data_dict"] = clients_data_dict
    data_logs["testing_data"] = (X_test, y_test)
    # if shared_pca:
    #   data_logs['shared_max_comps'] = shared_max_comps
    #   data_logs['shared_min_comps'] = shared_min_comps

    # Store the initial set of training parameters and data in a pickle file.
    # MODIFIED, layers: added is_qcnn to the name of the file
    if save_pkl:
      with open(f"{log_data_folder}/data_logs_n_samples_{n_samples}_dataset_type_{dataset_type}_classes_{'_'.join(classes)}_train_models_parallel_{train_models_parallel}_feature_skew_{feature_skew}_label_skew_{label_skew}_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_{generative}_qcnn_{is_qcnn}_{random_state}.pkl", "wb") as file:
        pickle.dump(data_logs, file)


    if use_torch:
      math_int = torch
    else:
      math_int = np

    # Maybe right here, have a torch data conversion function ??? (if not done in perform_federated_pca_mocked)
    clients_data_dict = convert_data_to_lib(clients_data_dict, math_int=math_int)
    if math_int == torch:
      # TODO, layers: convert y_test to a PyTorch tensor as well.
      X_test = torch.tensor(X_test, dtype=torch.float32)
      y_test = torch.tensor(y_test, dtype=torch.float32)
    print_cust(f"run_qfl_experiments_parallel_multiprocess, after torch conversion, clients_data_dict: {clients_data_dict}")

    if generative:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test.shape: {X_test.shape}")
      if len(X_test.shape) == 2:
        img_dim = math.isqrt(X_test.shape[1])
        assert (img_dim ** 2) == X_test.shape[1], f"run_qfl_experiments_parallel_multiprocess, X_test is not a perfect square, X_test.shape: {X_test.shape}"
        X_test = X_test.view(X_test.shape[0], img_dim, img_dim)
      print_cust(f"run_qfl_experiments_parallel_multiprocess, generative, X_test.max(): {X_test.max()}, X_test.min(): {X_test.min()}")
      save_tensors_to_folder(X_test, f"{targ_data_folder_prefix}", "img")
      n_samples_test = X_test.shape[0]


    # TODO: continue reading here, 7/9
    # also, please put this in a function......
    if shared_pca:
      # NOTE: this function MUTATES clients_data_dict
      pca_global, shared_min_comps, shared_max_comps, inv_pca_min, inv_pca_max = perform_federated_pca_mocked(clients_data_dict, max_size_clients, random_seed=random_state, math_int=math_int, device=device, generative=generative)
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
            # NOTE, layers: for less namespace conflicts, pca_info -> pca_info_cli
            (training_data, validation_data, pca_info_cli) = client_data
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
      pca_info = (pca_global, shared_min_comps, shared_max_comps, inv_pca_min, inv_pca_max, device)
    # NOTE, layers: continue reading here. (stopped here, 7/31)
    # DONE: TOMODIFY, layers: add an argument that specifies whether or not we need conv layers.
    # DONE: TOMODIFY, layers: do this in a torch.no_grad() context.
    # NO CHANGE (I believe the change should be made in the main call): TOMODIFY, layers: for additional layers for the same number of qubits, add them to the list IN ORDER (so the order of the block params really does matter)
    # (later, if you need to keep track of the layer information, you can do so using the qubits_and_layers_to_add_block_params arg for smaller sized models, so it is OK)
    with torch.no_grad():
      client_params_dict = initialize_client_params(clients_config_arg, model_size=min_size_clients, cur_client_params_dict=None, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params, train_models_parallel=train_models_parallel,
                                                    n_output_qubits=n_output_qubits, generative=generative, use_torch=use_torch, qnode_func=qnode_func, device=device, pca_info=pca_info, is_qcnn=is_qcnn, alt_zeros_init=alt_zeros_init)

    print_cust(f"run_qfl_experiments_parallel_multiprocess, first initialization of client params, client_params_dict: {client_params_dict}")

    # Make all parameters the same across clients with a new random initialization if we want clients to have the same initialization.
    if same_init:
      print_cust(f"run_qfl_experiments_parallel_multiprocess, same parameters initialization")
      with torch.no_grad():
        meta_params = generate_meta_params(client_params_dict, clients_data_dict, math_int=math_int)
        meta_params = generate_meta_params_random(meta_params, math_int=math_int)
        client_params_dict = broadcast_param_updates(client_params_dict, meta_params, math_int=math_int)
        print_cust(f"run_qfl_experiments_parallel_multiprocess, client_params_dict: {client_params_dict}")
        if initial_supp_params is not None:
          # client_params_dict = initial_supp_params
          for cli_type in client_params_dict:
            for cli_idx, cli_params in enumerate(client_params_dict[cli_type]):
              # DONE: TOMODIFY, layers: this is for the generative case; for the nongenerative case, have the code. should be pretty straightforward; just replace block params entirely
              # ^ and later, can do validation to make sure that the structure, order of the supplied block params is consistent with qubits_and_layers_to_add_block_params.
              if generative:
                cli_gen = cli_params[5][0]
                supp_cli_gen = initial_supp_params[cli_type][cli_idx][5][0]
                cli_gen.load_state_dict(supp_cli_gen.state_dict())
                cli_params[5][1].load_state_dict(initial_supp_params[cli_type][cli_idx][5][1].state_dict())
              elif not is_qcnn:
                cli_params_list = list(cli_params)
                cli_params_list[5] = initial_supp_params[cli_type][cli_idx][5]
                client_params_dict[cli_type][cli_idx] = tuple(cli_params_list)

          print_cust(f"run_qfl_experiments_parallel, supplied initial_supp_params, client_params_dict: {client_params_dict}")

    # Store the initial set of training parameters and data in a pickle file.
    print_cust(f"run_qfl_experiments_parallel, saving client_params_dict")
    # MODIFIED, layers: added is_qcnn to the name of the file
    if save_pkl:
      with open(f"{log_data_folder}/client_params_dict_n_samples_{n_samples}_dataset_type_{dataset_type}_classes_{'_'.join(classes)}_train_models_parallel_{train_models_parallel}_feature_skew_{feature_skew}_label_skew_{label_skew}_local_pca_{local_pca}_shared_pca_{shared_pca}_gen_{generative}_qcnn_{is_qcnn}_{random_state}.pkl", "wb") as file:
        pickle.dump(client_params_dict, file)

    # client_types = sorted(list(clients_config_arg.keys()))

    # Initialize the data logging information for each communication round.

    client_optims_dict = initialize_client_optimizers(client_params_dict, lr_gen, lr_disc, gen_betas, disc_betas, optim_type, existing_optims_dict=None, generative=generative, is_qcnn=is_qcnn, opt_layers=opt_layers)
    print_cust(f"run_qfl_experiments_parallel_multiprocess, client_optims_dict: {client_optims_dict}")

    for round_num in range(num_total_rounds):
      print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}")
      # Initialize the data logging information for this communication round.
      data_logs[round_num] = {}
      data_logs[round_num]["aggregated_params"] = None
      data_logs[round_num]["test_fid_score"] = None

      data_logs[round_num]["testing_acc"] = None
      data_logs[round_num]["testing_acc_stdev"] = None
      data_logs[round_num]["testing_acc_topk"] = None
      data_logs[round_num]["testing_loss"] = None

      if multiclassifier_type != "":
        data_logs[round_num]["testing_acc_classifiers"] = None
        data_logs[round_num]["testing_acc_stdev_classifiers"] = None
        data_logs[round_num]["testing_acc_topk_classifiers"] = None
        data_logs[round_num]["testing_loss_classifiers"] = None
        data_logs[round_num]["gen_all_probs"] = None

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
                                                        "testing_fid": None,
                                                        "grad_norms": None,
                                                        "optimizer_state": None,
                                                        "testing_acc_classifiers": None,
                                                        "testing_acc_stdev_classifiers": None,
                                                        "testing_acc_topk_classifiers": None,
                                                        "testing_loss_classifiers": None,
                                                        "gen_all_probs": None})

    # ---------- (UNCHANGED BOILERPLATE: data loading / setup) -------------
    # ... everything down to the initialisation of `client_params_dict`
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    #  Communication rounds – now PARALLEL
    # ---------------------------------------------------------------------
    # max_workers = None          # defaults to os.cpu_count()
    print_cust(f"run_qfl_experiments_parallel_multiprocess, os.cpu_count(): {os.cpu_count()}")
    if not generative:
      qnode_builder = create_qnode_qcnn      # tiny alias for pickling friendliness
      if multiclassifier_type == "multirun":
        # NOTE, depthFL: this is NOT a real qnode; its a caller that calls a qnode MULTIPLE times.
        qnode_builder = create_qnode_qcnn_multieval
      elif multiclassifier_type == "ancilla_endmeas":
        qnode_builder = create_qnode_qcnn_singleeval
      elif multiclassifier_type == "cheating":
        qnode_builder = create_qnode_qcnn_multieval_cheating
      elif multiclassifier_type == "tunnel_down":
        qnode_builder = create_qnode_qcnn_singleeval_tunneldown
    else:
      qnode_builder = create_qnode_qgan
    # DONE: TOMODIFY, depthFL: a function that is a wrapper around qnode_builder that yields a custom QNode.
    # Q: am I using it as the qnode_builder then? so I'm just replacing it? A: yes, I think so.

    print_cust(f"run_qfl_experiments_parallel_multiprocess, qnode_builder: {qnode_builder}")

    rounds_elapsed = 0
    for cur_shared_model_size, model_size_rounds in client_types_to_rounds.items():

      if cur_shared_model_size > min_size_clients and not train_models_parallel:
        # DONE: TOMODIFY, layers: do this in a torch.no_grad() context.
        with torch.no_grad():
          client_params_dict = initialize_client_params(clients_config_arg, model_size=cur_shared_model_size, cur_client_params_dict=client_params_dict, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params, train_models_parallel=train_models_parallel,
                                                    n_output_qubits=n_output_qubits, generative=generative, use_torch=use_torch, qnode_func=qnode_func, device=device, pca_info=pca_info, is_qcnn=is_qcnn, alt_zeros_init=alt_zeros_init)
        # DONE: TOMODIFY, layers: inject the mask_grads argument here to specify what params to optimize over.
        print_cust(f"run_qfl_experiments_parallel_multiprocess, after expansion reinitialization, client_params_dict: {client_params_dict}")

        client_optims_dict = initialize_client_optimizers(client_params_dict, lr_gen, lr_disc, gen_betas, disc_betas, optim_type, existing_optims_dict=client_optims_dict, generative=generative, is_qcnn=is_qcnn, opt_layers=opt_layers)

        existing_clients_datadict = list(clients_data_dict.keys())
        for existing_client_size in existing_clients_datadict:
          if cur_shared_model_size > existing_client_size:
            del clients_data_dict[existing_client_size]

        print_cust(f"run_qfl_experiments_parallel_multiprocess, not train_models_parallel, clients_data_dict.keys(): {clients_data_dict.keys()}")

      for round_num in range(model_size_rounds):
          round_num += rounds_elapsed
          print_cust(f"run_qfl_experiments_parallel_multiprocess, [round {round_num}] starting")

          # TODO: DI in the right workers and args for mp_context based on processpoolexecutor or threadpoolexecutor.
          # NOTE, depthFL: with multiple clients, trying dummyexecutor for now.
          with DummyExecutor(max_workers=max_workers) as pool:
              print_cust(f"run_qfl_experiments_parallel_multiprocess, max_workers: {max_workers}")
              futures = []

              # -------------------------------------------------------------
              # spawn one worker job per client
              # -------------------------------------------------------------
              for client_type, client_params in client_params_dict.items():
                  client_data_list = clients_data_dict[client_type]
                  # NOCHANGE: TOMODIFY, depthFL: if train_models_parallel and not is_qcnn, dynamically figure out the client size based on the max number
                  # of qubits needed to run their block params.
                  # ^ OR, can just not change it for now; this allows different client types to train for different local epochs.
                  if train_models_parallel:
                    cur_model_size   = client_type
                  else:
                    cur_model_size   = cur_shared_model_size
                  print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_model_size: {cur_model_size}")
                  # DONE: TOMODIFY, layers: inject the arg here to not do QCNN and thus have conv_layers be 0.
                  conv_layers      = compute_conv_layers(cur_model_size,
                                                        n_output_qubits, generative=generative, is_qcnn=is_qcnn)
                  print_cust(f"run_qfl_experiments_parallel_multiprocess, conv_layers: {conv_layers}")
                  num_local_epochs = clients_config_arg[cur_model_size]["local_epochs"]

                  for client_idx, client_params_indiv in enumerate(client_params):
                      client_train_data = client_data_list[client_idx][0]
                      client_val_data   = client_data_list[client_idx][1]

                      print_cust(f"run_qfl_experiments_parallel_multiprocess, client_train_data: {client_train_data}, client_val_data: {client_val_data}")

                      grad_mask = None

                      # DONE: TOMODIFY, layers: can temporarily override this to not call mask_gradients; doing that in my optimizer construction.
                      if mask_grads:
                        if cur_model_size > min_size_clients and not generative and is_qcnn:
                          print_cust(f"run_qfl_experiments_parallel_multiprocess, calling mask_gradients()")
                          grad_mask = mask_gradients(client_params_indiv)

                      # Create the feature list indicating what features for which qubits are to be used for this client, as well as the expansion data (used for identity
                      # initialization).
                      # expand = False
                      # if cur_model_size > min_size_clients:
                      #   expand = True
                      # feature_list, expansion_data = create_feat_list_expansion_data(cur_model_size, conv_layers, expand=expand, pool_in=pool_in, min_qubits_noexpand=min_size_clients,
                      #                                                               train_models_parallel=train_models_parallel, feat_sel_type=feat_sel_type)

                      # build client‑specific testing data once here
                      if local_pca and not generative:
                        # NOTE: why was client_data_indiv used here?? doesn't make that much sense.
                        # NOTE, layers: changed client_pca_info here to use index 2 instead of 1; not sure if it's right
                        client_pca_info = client_data_list[client_idx][2]
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, client_pca_info: {client_pca_info}")
                        client_pca = client_pca_info[0]
                        client_data_pca = client_pca_info[1]
                        # DONE: TOMODIFY, layers: note that the PCA is for numpy, and X_test is a pytorch tensors, so I'll need to do some data conversions (applies for this entire block of code below)
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test.shape: {X_test.shape}, type(X_test): {type(X_test)}")
                        if use_torch:
                          X_test_np = X_test.detach().cpu().numpy()
                        else:
                          X_test_np = X_test
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_np.shape: {X_test_np.shape}, type(X_test_np): {type(X_test_np)}")
                        X_test_client_pca = client_pca.transform(X_test_np)
                        # Scale each PCA component independently to [0, π]
                        # MODIFIED, layers: changed np -> math_int
                        if use_torch:
                          # MODIFIED, layers: converted to torch after PCA transform
                          # NOTE, layers: I might get dtype issues here, in which case I'd need to change to torch.float32 explicitly.
                          X_test_client_pca = torch.from_numpy(X_test_client_pca).to(device)
                        X_test_client_angle = math_int.zeros_like(X_test_client_pca)
                        # assuming that the number of components is simply the client's data type
                        # TODO: rescale the testing data to match the PCA scale for each client
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_client_pca.shape: {X_test_client_pca.shape}, type(X_test_client_pca): {type(X_test_client_pca)}")
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}")
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, shared_min_comps: {shared_min_comps}, shared_max_comps: {shared_max_comps}, type(shared_min_comps): {type(shared_min_comps)}, type(shared_max_comps): {type(shared_max_comps)}")
                        for i in range(cur_model_size):
                            comp = X_test_client_pca[:, i]
                            if shared_pca:
                              # NOTE, layers: technically I should do instanceof check here to make sure that
                              # these are torch return types
                              lo, hi = shared_min_comps.values[i], shared_max_comps.values[i]
                              print_cust(f"run_qfl_experiments_parallel_multiprocess, comp.shape: {comp.shape}, type(comp): {type(comp)}, lo.shape: {lo.shape}, hi.shape: {hi.shape}, type(lo): {type(lo)}, type(hi): {type(hi)}")
                              comp_norm = ( (comp - lo) / (hi - lo + 1e-8) )
                              # NOTE, depthFL: need to clip comp_norm????
                              # comp_norm = np.clip(comp_norm, 0, 1)
                            else:
                              orig_comp = client_data_pca[:, i]
                              comp_norm = (comp - orig_comp.min()) / (orig_comp.max() - orig_comp.min() + 1e-8)
                            # MODIFIED, layers: changed np -> math_int
                            X_test_client_angle[:, i] = comp_norm * math_int.pi
                      else:
                        X_test_client_angle = X_test

                      # # Select the features for amplitude encoding based on feat_ordering.
                      # if amp_embed:
                      #   if feat_ordering == "highest_var":
                      #     variances = X_angles.var(axis=0, ddof=0)
                      #     order = np.argsort(variances)[::-1]
                      #     X_test_client_angle = X_test_client_angle[:, order] + 1e-3
                      #   testing_data = reorder_amplitude_data(X_test_client_angle, feature_list)
                      # else:
                      #   testing_data = X_test_client_angle[:, feature_list]

                      print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
                      print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")
                      if local_pca and not generative:
                        X_test_client_angle = math_int.clip(X_test_client_angle, 0.0, math_int.pi)
                      print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
                      print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")
                      testing_data = (X_test_client_angle, y_test)


                      if generative:
                        generator_metadata = client_params_indiv[5][0].get_data_components()
                        disc_metadata = client_params_indiv[5][1].get_data_components()
                        client_params_indiv_serialized = [[] for _ in range(len(client_params_indiv))]
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, len(client_params_indiv_serialized): {len(client_params_indiv_serialized)}")
                        client_params_indiv_serialized[5].append((client_params_indiv[5][0].state_dict(), generator_metadata))
                        client_params_indiv_serialized[5].append((client_params_indiv[5][1].state_dict(), disc_metadata))
                        # client_params_indiv[5][0] = (client_params_indiv[5][0].state_dict(), generator_metadata)
                        # client_params_indiv[5][1] = (client_params_indiv[5][1].state_dict(), disc_metadata)
                        # print_cust(f"run_qfl_experiments_parallel_multiprocess, 'serializing' model, client_params_indiv: {client_params_indiv}")
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, 'serializing' model, client_params_indiv_serialized: {client_params_indiv_serialized}")
                        client_params_indiv = client_params_indiv_serialized

                        client_optims_indiv = client_optims_dict[client_type][client_idx]

                        client_optims_indiv_serialized = [[] for _ in range(len(client_optims_indiv))]

                        client_optims_indiv_serialized[5].append(client_optims_indiv[5][0].state_dict())

                        client_optims_indiv_serialized[5].append(client_optims_indiv[5][1].state_dict())

                        # client_optims_indiv_serialized = [client_optims_indiv[5][0].state_dict(), client_optims_indiv[5][1].state_dict()]

                      # DONE: TOMODIFY, layers: for the client_optims_indiv_serialized, do something similar in the new format of the optims dictionary for
                      # the classifier case.

                      # NO CHANGE: TOMODIFY, layers: for parameters that don't have a value (None), from the generative case, set them to be None.
                      # DONE: TOMODIFY, layers: for this overall function, have an argument to specify the pennylane_interface (for layers, I want it to be torch).
                      elif not is_qcnn:
                        client_params_indiv_serialized = copy.deepcopy(client_params_indiv)
                        client_optims_indiv = client_optims_dict[client_type][client_idx]

                        client_optims_indiv_serialized = [None for _ in range(len(client_optims_indiv))]

                        client_optims_indiv_serialized[5] = client_optims_indiv[5].state_dict()

                        print_cust(f"run_qfl_experiments_parallel, not is_qcnn, client_params_indiv_serialized: {client_params_indiv_serialized}")
                        print_cust(f"run_qfl_experiments_parallel, not is_qcnn, client_optims_indiv_serialized: {client_optims_indiv_serialized}")

                      if client_type in qubits_and_layer_types_block_params:
                        layer_types_list = qubits_and_layer_types_block_params[client_type]
                      else:
                        layer_types_list = []
                      
                      print_cust(f"run_qfl_experiments_parallel_multiprocess, layer_types_list: {layer_types_list}")

                      job = dict(client_type=client_type,
                                client_idx=client_idx,
                                client_params_indiv=client_params_indiv,
                                client_train_data=client_train_data,
                                client_val_data=client_val_data,
                                testing_data=testing_data,
                                cur_model_size=cur_model_size,
                                min_size_clients=min_size_clients,
                                pool_in=pool_in,
                                feat_sel_type=feat_sel_type,
                                train_models_parallel=train_models_parallel,
                                amp_embed=amp_embed, shots=shots,
                                local_batch_size=local_batch_size,
                                local_lr=local_lr,
                                num_local_epochs=num_local_epochs,
                                conv_layers=conv_layers,
                                feat_ordering=feat_ordering,
                                classes=classes,
                                qnode_builder=qnode_builder,
                                num_total_rounds=num_total_rounds,
                                round_num=round_num,
                                grad_mask=grad_mask,
                                generative=generative,
                                lr_gen=lr_gen,
                                lr_disc=lr_disc,
                                noise_func=noise_func,
                                criterion_func=criterion_func,
                                log_data_folder=log_data_folder,
                                device=device,
                                client_optims_indiv=client_optims_indiv_serialized,
                                optim_type=optim_type,
                                gen_betas=gen_betas,
                                disc_betas=disc_betas,
                                use_torch=use_torch,
                                pennylane_interface=pennylane_interface,
                                opt_layers=opt_layers,
                                layer_types_list=layer_types_list,
                                loss_type=loss_type,
                                lr_disc_decay=lr_disc_decay,
                                cont_optim_state=cont_optim_state)
                      futures.append(pool.submit(_train_single_client, job))

              # -------------------------------------------------------------
              #  gather results as they finish
              # -------------------------------------------------------------
              for fut in as_completed(futures):
                  res = fut.result()

                  ctype, cidx = res["client_type"], res["client_idx"]
                  print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, ctype: {ctype}, cidx: {cidx}")

                  if not generative and is_qcnn:
                    client_params_dict[ctype][cidx] = res["trained_params"]

                    # write metrics into the log
                    dlog = data_logs[round_num][ctype]["client_metrics"][cidx]
                    dlog["trained_params"]   = res["trained_params"]
                    dlog["minibatch_losses"] = res["minibatch_losses"]
                    dlog["validation_losses"] = res["validation_losses"]
                    dlog["training_acc"]     = res["train_acc"]
                    dlog["testing_acc"]      = res["test_acc"]
                    dlog["testing_loss"]     = res["test_loss"]
                    dlog["training_acc_stdev"] = res["train_acc_stdev"]
                    dlog["testing_acc_stdev"]  = res["test_acc_stdev"]
                    dlog["training_acc_topk"]  = res["train_acc_topk"]
                    dlog["testing_acc_topk"]   = res["test_acc_topk"]

                    print_cust(f'run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, cur_model_size: {cur_model_size}, client_idx: {cidx}, train_acc: {res["train_acc"]}, test_acc: {res["test_acc"]}, test_loss: {res["test_loss"]}, train_acc_stdev: {res["train_acc_stdev"]}, test_acc_stdev: {res["test_acc_stdev"]}, train_acc_topk: {res["train_acc_topk"]}, test_acc_topk: {res["test_acc_topk"]}')
                  elif not generative:
                    # DONE: TOMODIFY, layers: in addition, if optimizer state is supplied, then update the state dict for the optimizer here.
                    # (for debug, for layers, can later store the optimizer state at each round.)
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, res['trained_disc_params']: {res['trained_disc_params']}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, qnode_builder: {qnode_builder}")
                    cur_cli_params = client_params_dict[ctype][cidx]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_params[5]: {cur_cli_params[5]}")
                    cur_cli_params_list = list(cur_cli_params)
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_params_list: {cur_cli_params_list}")
                    cur_cli_params_list[5] = res["trained_disc_params"]
                    cur_cli_params = tuple(cur_cli_params_list)
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, type(cur_cli_params): {type(cur_cli_params)}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_params[5]: {cur_cli_params[5]}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_params: {cur_cli_params}")
                    # MODIFIED, depthFL: update client params.
                    client_params_dict[ctype][cidx] = cur_cli_params

                    for i, p in enumerate(cur_cli_params[5]):
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_params[5][{i}]:\n{p}")

                    # BUG (suspected): NOT changing client_params_dict.
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, client_params_dict[ctype][cidx]: {client_params_dict[ctype][cidx]}")

                    cur_cli_optims = client_optims_dict[ctype][cidx]
                    cur_cli_optim_classifier = cur_cli_optims[5]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_classifier: {cur_cli_optim_classifier}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_classifier.state_dict(): {cur_cli_optim_classifier.state_dict()}")
                    # NOTE: loads in a SHALLOW copy of state dict for the classifier optimizer.
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, res['trained_optim_disc']: {res['trained_optim_disc']}")
                    cur_cli_optim_classifier.load_state_dict(res["trained_optim_disc"])
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_classifer: {cur_cli_optim_classifier}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_classifier.state_dict(): {cur_cli_optim_classifier.state_dict()}")

                    train_metrics_dict = res["train_metrics_dict"]

                    print_cust(f"run_qfl_experiments_multiprocess, train_metrics_dict: {train_metrics_dict}")

                    dlog = data_logs[round_num][ctype]["client_metrics"][cidx]

                    dlog["trained_params"] = [copy.deepcopy(res["trained_disc_params"])]
                    dlog["minibatch_losses"] = [train_metrics_dict["disc_loss"]]
                    dlog["grad_norms"] = [train_metrics_dict["disc_grad_norms"]]
                    # how in the world is this sending a shallow copy????????
                    # it is. that's ok
                    dlog["optimizer_state"] = [copy.deepcopy(res["trained_optim_disc"])]

                  else:
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, res['trained_gen_params']: {res['trained_gen_params']}, res['trained_disc_params']: {res['trained_disc_params']}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, qnode_builder: {qnode_builder}")
                    cur_cli_params = client_params_dict[ctype][cidx]
                    cur_cli_gen = cur_cli_params[5][0]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_gen: {cur_cli_gen}")
                    # constructed_cli_gen = build_patchquantumgen(cur_cli_gen_metadata, res["trained_gen_params"], qnode_builder)
                    cur_cli_disc = cur_cli_params[5][1]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_disc: {cur_cli_disc}")
                    # constructed_cli_disc = build_pca_discriminator(cur_cli_disc_metadata, res["trained_disc_params"])

                    cur_cli_gen.load_state_dict(res["trained_gen_params"])
                    cur_cli_disc.load_state_dict(res["trained_disc_params"])
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_gen: {cur_cli_gen}")
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_disc: {cur_cli_disc}")

                    cur_cli_optims = client_optims_dict[ctype][cidx]
                    cur_cli_optim_gen = cur_cli_optims[5][0]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_gen: {cur_cli_optim_gen}")
                    cur_cli_optim_gen.load_state_dict(res["trained_optim_gen"])
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_gen: {cur_cli_optim_gen}")

                    cur_cli_optim_disc = cur_cli_optims[5][1]
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_disc: {cur_cli_optim_disc}")
                    cur_cli_optim_disc.load_state_dict(res["trained_optim_disc"])
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_optim_disc: {cur_cli_optim_disc}")
                    # cur_cli_params[5][0] = constructed_cli_gen
                    # cur_cli_params[5][1] = constructed_cli_disc
                    # can print out client_params_dict to verify that it changes here if I want I guess
                    # print_cust(f"run_qfl_experiments_parallel_multiprocess, cur_cli_p")

                    train_metrics_dict = res["train_metrics_dict"]

                    print_cust(f"run_qfl_experiments_multiprocess, train_metrics_dict: {train_metrics_dict}")

                    dlog = data_logs[round_num][ctype]["client_metrics"][cidx]

                    print_cust(f"run_qfl_experiments_multiprocess, res['trained_optim_gen']: {res['trained_optim_gen']}")
                    print_cust(f"run_qfl_experiments_multiprocess, res['trained_optim_disc']: {res['trained_optim_disc']}")

                    test_gen_opt_param_log = res['trained_optim_gen']['state'][0]['exp_avg']
                    test_disc_opt_param_log = res['trained_optim_disc']['state'][0]['exp_avg']

                    print_cust(f"run_qfl_experiments_multiprocess, test_gen_opt_param_log: {test_gen_opt_param_log}")
                    print_cust(f"run_qfl_experiments_multiprocess, test_disc_opt_param_log: {test_disc_opt_param_log}")

                    print_cust(f"run_qfl_experiments_multiprocess, test_gen_opt_param_log.storage().is_shared(): {test_gen_opt_param_log.storage().is_shared()}")
                    print_cust(f"run_qfl_experiments_multiprocess, test_disc_opt_param_log.storage().is_shared(): {test_disc_opt_param_log.storage().is_shared()}")
                    print_cust(f"run_qfl_experiments_multiprocess, test_gen_opt_param_log.data_ptr(): {test_gen_opt_param_log.data_ptr()}")
                    print_cust(f"run_qfl_experiments_multiprocess, test_disc_opt_param_log.data_ptr(): {test_disc_opt_param_log.data_ptr()}")

                    dlog["trained_params"] = [res["trained_gen_params"], res["trained_disc_params"]]
                    dlog["minibatch_losses"] = [train_metrics_dict["gen_loss"], train_metrics_dict["disc_loss"]]
                    dlog["grad_norms"] = [train_metrics_dict["gen_grad_norms"], train_metrics_dict["disc_grad_norms"]]
                    # how in the world is this sending a shallow copy????????
                    # it is. that's ok
                    dlog["optimizer_state"] = [copy.deepcopy(res["trained_optim_gen"]), copy.deepcopy(res["trained_optim_disc"])]
                    # dlog["validation_losses"] = res["validation_losses"]
                    # dlog["training_acc"]     = res["train_acc"]
                    # dlog["testing_acc"]      = res["test_acc"]
                    # dlog["testing_loss"]     = res["test_loss"]
                    # dlog["training_acc_stdev"] = res["train_acc_stdev"]
                    # dlog["testing_acc_stdev"]  = res["test_acc_stdev"]
                    # dlog["training_acc_topk"]  = res["train_acc_topk"]
                    # dlog["testing_acc_topk"]   = res["test_acc_topk"]






          # -----------------------------------------------------------------
          #  aggregation (unchanged)
          # -----------------------------------------------------------------

          print_cust(f"run_qfl_experiments_parallel_multiprocess, before aggregation and broadcasting, round_num: {round_num}, client_params_dict: {client_params_dict}")

          print_cust(f"run_qfl_experiments_parallel_multiprocess, parameter aggregation, agg_strategy: {agg_strategy}, morepers: {morepers}")
          if agg_strategy == "fedavg":
              aggregated_params = federated_averaging(client_params_dict,
                                                      clients_data_dict)
          elif agg_strategy == "fedavg_quat":
              aggregated_params = federated_averaging_quat(client_params_dict,
                                                          clients_data_dict)
          elif agg_strategy == "fedavg_circ":
              aggregated_params = federated_averaging_circular_parallel_shared(
                                    client_params_dict, clients_data_dict, generative=generative)
          else:
              raise ValueError(f"unknown agg_strategy {agg_strategy}")

          if morepers == "aggshared":
              client_params_dict = broadcast_param_updates_shared(
                                      client_params_dict, aggregated_params, math_int=math_int)
          elif morepers == "aggnoconv":
              client_params_dict = broadcast_param_updates_shared_noconv(
                                      client_params_dict, aggregated_params)
          elif morepers == "aggnoconvnofinal":
              client_params_dict = broadcast_param_updates_shared_noconv_nofinal(
                                      client_params_dict, aggregated_params)
          else:
            print_cust(f"run_qfl_experiments_parallel, no broadcasting performed, morepers: {morepers}")

          # Store the aggregated parameters for this round.
          # TODO: aggregated parameters don't have a significant meaning as not all of them are broadcasted to clients based on the broadcasting/personalization scheme
          data_logs[round_num]["aggregated_params"] = aggregated_params

          print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, aggregated_params: {aggregated_params}")
          print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, client_params_dict: {client_params_dict}")

          # TODO: recompute metrics of the SHARED model for EACH client.
          # BUGFIX, depthFL: changed one_cli_size to be the maximum sized client.
          # NOTE: this assumes that the MAXIMUM "qubits" client has the MOST layers.
          one_cli_size = max(list(client_params_dict.keys()))

          print_cust(f"run_qfl_experiments_parallel_multiprocess, one_cli_size: {one_cli_size}")

          if generative:
            # NOTE: this assumes that all generators and discriminators have the same parameters.
            agg_model = client_params_dict[one_cli_size][0][5][0]
            # agg_disc = client_params_dict[one_cli_size][0][5][1]
          else:
            agg_model = client_params_dict[one_cli_size][0]


          test_acc, test_acc_stdev, test_acc_topk, test_loss, test_fid_score = (None, None, None, None, None)

          if fid_batch_size is None and generative:
            fid_batch_size = n_samples_test

          if not generative and not is_qcnn:
            # NOTE, layers: only doing this for the not is_qcnn case.
            # TOMODIFY, layers: change cur_shared_model_size here. and also, subset the testing data here.
            num_qubits_bps = []
            for block_param in agg_model[5]:
              num_qubits_bps.append(block_param.shape[1])

            print_cust(f"run_qfl_experiments_parallel_multiprocess, not generative and not is_qcnn, num_qubits_bps: {num_qubits_bps}")

            # NOTE, layers: this is a logical override.
            cur_shared_model_size = max(num_qubits_bps)
            if local_pca or shared_pca:
              print_cust(f"run_qfl_experiments_parallel_multiprocess, not generative and not is_qcnn, cur_shared_model_size: {cur_shared_model_size}")
              X_test_client_angle = X_test_client_angle[:, :cur_shared_model_size]
            # TOMODIFY, depthFL: change the name of this variable (for key into qubits_and_layer_types_block_params)
            largest_clisize_layertypes = max(qubits_and_layer_types_block_params)
            print_cust(f"run_qfl_experiments_parallel_multiprocess, largest_clisize_layertypes: {largest_clisize_layertypes}")
            layer_types_list_largest = qubits_and_layer_types_block_params[largest_clisize_layertypes]
            # TOMODIFY, layers: supply layer_types_list to this qnode_builder
            qnode_test = qnode_builder(cur_shared_model_size, conv_layers, [], n_classes=len(classes), pennylane_interface=pennylane_interface, layer_types_list=layer_types_list_largest)

            print_cust(f"run_qfl_experiments_parallel_multiprocess, X_test_client_angle: {X_test_client_angle}")
            print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.shape: {X_test_client_angle.shape}, type(X_test_client_angle): {type(X_test_client_angle)}, y_test.shape: {y_test.shape}, type(y_test): {type(y_test)}")
            print_cust(f"run_qfl_experiments_parallel, X_test_client_angle.min(): {X_test_client_angle.min()}, X_test_client_angle.max(): {X_test_client_angle.max()}, y_test.min(): {y_test.min()}, y_test.max(): {y_test.max()}")

          # TOMODIFY, layers: set this to be > 101, so that I always compute testing accuracy for logging.
          print_cust(f"run_qfl_experiments_parallel_multiprocess, testacc_rd_cutoff: {testacc_rd_cutoff}")
          if num_total_rounds > testacc_rd_cutoff:
            if (round_num + 1) % 10 == 0:
              if not generative:
                # TODO: this is broken for the non-generative case. process the testing data outside of the loop for each client.. not doing that now tho; not too relevant for
                # generative case
                # TODO: passing in testing data for multiproc seems to be broken here for the nongenerative case. can try to fix later.
                # TOMODIFY, layers (and in general, for classification): testing_data is outside of scope here. Create a 'global' set of testing data (this assumes that
                # testing data is independent of any clients' PCA). Make it clear what testing data I'm using.
                # ^ quickly hacked this by using out of scope testing data for quick impl... but can recreate the testing data as well
                # DONE: TOMODIFY, layers: call this in a torch.no_grad() context.
                # NOTE: for is_qcnn case, I think I override the testing_acc here (check with ctrl + f "testing_acc" ?)
                with torch.no_grad():
                  if morepers != "mocked_bcast":
                    if multiclassifier_type == "":
                      test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(agg_model, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                    else:
                      test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = compute_metrics_angle_param_batch(agg_model, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                    data_logs[round_num]["testing_acc"] = test_acc
                    data_logs[round_num]["testing_acc_stdev"] = test_acc_stdev
                    data_logs[round_num]["testing_acc_topk"] = test_acc_topk
                    data_logs[round_num]["testing_loss"] = test_loss
                    if multiclassifier_type != "":
                      data_logs[round_num]["testing_acc_classifiers"] = avg_acc_classifiers
                      data_logs[round_num]["testing_acc_stdev_classifiers"] = std_acc_classifiers
                      data_logs[round_num]["testing_acc_topk_classifiers"] = top_k_accuracies_classifiers
                      data_logs[round_num]["testing_loss_classifiers"] = avg_loss_classifiers
                      data_logs[round_num]["gen_all_probs"] = gen_all_probs
                  else:
                    print_cust(f"run_qfl_experiments_parallel_multiprocess, morepers is mocked_bcast, morepers: {morepers}")
                    # compute the testing acc for each client.
                    for cli_type, cli_params_list in client_params_dict.items():
                      for cli_idx, cli_params in enumerate(cli_params_list):
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, cli_type: {cli_type}, cli_idx: {cli_idx}")
                        print_cust(f"run_qfl_experiments_parallel_multiprocess, cli_params: {cli_params}")
                        if multiclassifier_type == "":
                          test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(cli_params, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                        else:
                          test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = compute_metrics_angle_param_batch(cli_params, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                        sel_dlog = data_logs[round_num][cli_type]["client_metrics"][cli_idx]
                        sel_dlog["testing_acc"] = test_acc
                        sel_dlog["testing_acc_stdev"] = test_acc_stdev
                        sel_dlog["testing_acc_topk"] = test_acc_topk
                        sel_dlog["testing_loss"] = test_loss
                        if multiclassifier_type != "":
                          sel_dlog["testing_acc_classifiers"] = avg_acc_classifiers
                          sel_dlog["testing_acc_stdev_classifiers"] = std_acc_classifiers
                          sel_dlog["testing_acc_topk_classifiers"] = top_k_accuracies_classifiers
                          sel_dlog["testing_loss_classifiers"] = avg_loss_classifiers
                          sel_dlog["gen_all_probs"] = gen_all_probs
              else:
                if compute_fid:
                  print_cust(f"run_qfl_experiments_multiprocess, computing FID")
                  folder_id_suffix = f"round_num_{round_num}_cur_shared_model_size_{cur_shared_model_size}"
                  test_fid_score = compute_fid_to_data(agg_model, noise_func, f"{targ_data_folder_prefix}",
                                                      f"{gen_data_folder_prefix}/{folder_id_suffix}", n_samples_test, device, fid_batch_size=fid_batch_size, resc_invpca=resc_invpca)
                  data_logs[round_num]["test_fid_score"] = test_fid_score
            else:
              test_acc, test_acc_stdev, test_acc_topk, test_loss = (None, None, None, None)
          else:
            if not generative:
              # TOMODIFY, layers (and in general, for classification): testing_data is outside of scope here. Create a 'global' set of testing data (this assumes that
              # testing data is independent of any clients' PCA). Make it clear what testing data I'm using.
              # ^ quickly hacked this by using out of scope testing data for quick impl... but can recreate the testing data as well
              # DONE: TOMODIFY, layers: call this in a torch.no_grad() context.
              # NOTE: for is_qcnn case, I think I override the testing_acc here (check with ctrl + f "testing_acc" ?)
              with torch.no_grad():
                if morepers != "mocked_bcast":
                  if multiclassifier_type == "":
                    test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(agg_model, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                  else:
                    test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = compute_metrics_angle_param_batch(agg_model, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                  data_logs[round_num]["testing_acc"] = test_acc
                  data_logs[round_num]["testing_acc_stdev"] = test_acc_stdev
                  data_logs[round_num]["testing_acc_topk"] = test_acc_topk
                  data_logs[round_num]["testing_loss"] = test_loss
                  if multiclassifier_type != "":
                    data_logs[round_num]["testing_acc_classifiers"] = avg_acc_classifiers
                    data_logs[round_num]["testing_acc_stdev_classifiers"] = std_acc_classifiers
                    data_logs[round_num]["testing_acc_topk_classifiers"] = top_k_accuracies_classifiers
                    data_logs[round_num]["testing_loss_classifiers"] = avg_loss_classifiers
                    data_logs[round_num]["gen_all_probs"] = gen_all_probs
                else:
                  print_cust(f"run_qfl_experiments_parallel_multiprocess, morepers is mocked_bcast, morepers: {morepers}")
                  # compute the testing acc for each client.
                  for cli_type, cli_params_list in client_params_dict.items():
                    for cli_idx, cli_params in enumerate(cli_params_list):
                      print_cust(f"run_qfl_experiments_parallel_multiprocess, cli_type: {cli_type}, cli_idx: {cli_idx}")
                      print_cust(f"run_qfl_experiments_parallel_multiprocess, cli_params: {cli_params}")
                      if multiclassifier_type == "":
                        test_acc, test_acc_stdev, test_acc_topk, test_loss = compute_metrics_angle_param_batch(cli_params, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                      else:
                        test_acc, test_acc_stdev, test_acc_topk, test_loss, avg_acc_classifiers, std_acc_classifiers, top_k_accuracies_classifiers, avg_loss_classifiers, gen_all_probs = compute_metrics_angle_param_batch(cli_params, X_test_client_angle, y_test, layers=conv_layers, shots=shots, batch_size=local_batch_size, qnode=qnode_test, math_int=math_int)
                      sel_dlog = data_logs[round_num][cli_type]["client_metrics"][cli_idx]
                      sel_dlog["testing_acc"] = test_acc
                      sel_dlog["testing_acc_stdev"] = test_acc_stdev
                      sel_dlog["testing_acc_topk"] = test_acc_topk
                      sel_dlog["testing_loss"] = test_loss
                      if multiclassifier_type != "":
                        sel_dlog["testing_acc_classifiers"] = avg_acc_classifiers
                        sel_dlog["testing_acc_stdev_classifiers"] = std_acc_classifiers
                        sel_dlog["testing_acc_topk_classifiers"] = top_k_accuracies_classifiers
                        sel_dlog["testing_loss_classifiers"] = avg_loss_classifiers
                        sel_dlog["gen_all_probs"] = gen_all_probs
            else:
              if compute_fid:
                print_cust(f"run_qfl_experiments_multiprocess, computing FID")
                folder_id_suffix = f"round_num_{round_num}_cur_shared_model_size_{cur_shared_model_size}"
                test_fid_score = compute_fid_to_data(agg_model, noise_func, f"{targ_data_folder_prefix}",
                                                    f"{gen_data_folder_prefix}/{folder_id_suffix}", n_samples_test, device, fid_batch_size=fid_batch_size, resc_invpca=resc_invpca)
                data_logs[round_num]["test_fid_score"] = test_fid_score
          if not generative:
            print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, cur_model_size: {cur_model_size}, test_acc: {test_acc}, test_loss: {test_loss}, test_acc_stdev: {test_acc_stdev}, test_acc_topk: {test_acc_topk}")
          else:
            print_cust(f"run_qfl_experiments_parallel_multiprocess, round_num: {round_num}, cur_model_size: {cur_model_size}, test_fid_score: {test_fid_score}")

          print_cust(f"run_qfl_experiments_parallel_multiprocess, [round {round_num}] done\n")
      rounds_elapsed += model_size_rounds

    return data_logs