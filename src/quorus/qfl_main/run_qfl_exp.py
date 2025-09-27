"""## QFL Experiment Function"""

import math
import pickle

from quorus.data_ops.data_loading_wrapper import load_dataset
from quorus.logging.custom_slog import print_cust
from quorus.metrics_funcs.acc_funcs.accuracy_func import compute_avg_acc_angle_param_batch
from quorus.metrics_funcs.pennylane_lossfns.pennylane_lossfn_batch import compute_loss_angle_param_batch
from quorus.misc_utils.create_feat_list_expansion_data import create_feat_list_expansion_data
from quorus.qfl_utils.aggregation_funcs.federated_averaging import federated_averaging
from quorus.qfl_utils.aggregation_funcs.federated_averaging_circular import federated_averaging_circular
from quorus.qfl_utils.aggregation_funcs.federated_averaging_quat import federated_averaging_quat
from quorus.qfl_utils.cli_params_init.initialize_client_params import initialize_client_params
from quorus.qfl_utils.data_splitting.split_data_federated import split_data_federated
from quorus.qnode_funcs.qnode_creation_funcs.create_qnode_qcnn import create_qnode_qcnn
from quorus.training_funcs.pennylane_training.mask_gradients import mask_gradients
from quorus.training_funcs.pennylane_training.train_mod_pennylane import train_epochs_angle_param_adam
from pennylane import numpy as np

def run_qfl_experiments(clients_config_arg, classes=["4", "9"], n_samples=1000, dataset_type="mnist", agg_strategy="fedavg", test_frac=0.2, val_frac=0.1, random_state=42, pool_in=True,
                        local_batch_size=32, local_lr=0.01, shots=1024, debug=False, init_client_data_dict=None, save_pkl=False, mask_grads=False, qubits_and_layers_to_add_block_params=[],
                        train_models_parallel=False):
  max_size_clients = max(clients_config_arg.keys())
  min_size_clients = min(clients_config_arg.keys())
  X_angles, y = load_dataset(dataset_type=dataset_type, classes=classes, n_samples=n_samples, num_feats=max_size_clients)

  print_cust(f"run_qfl_experiments, X_angles.shape: {X_angles.shape}")

  if init_client_data_dict is not None:
    clients_data_dict = init_client_data_dict["clients_data_dict"]
    (X_test, y_test) = init_client_data_dict["testing_data"]
    print_cust(f"run_qfl_experiments, loaded in existing data")
  else:
    clients_data_dict, (X_test, y_test) = split_data_federated(X_angles, y, clients_config_arg, test_frac, val_frac=val_frac, random_state=random_state)
    print_cust(f"run_qfl_experiments, generated new data")

  if debug:
    for client_type in clients_data_dict:
      clients_data_list = clients_data_dict[client_type]
      print_cust(f"client_type: {client_type}, len(clients_data_list): {len(clients_data_list)}")
      for client_idx in range(len(clients_data_list)):
        client_data = clients_data_list[client_idx]
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, len(client_data): {len(client_data)}")
        (training_data, validation_data) = client_data
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, len(training_data): {len(training_data)}, len(validation_data): {len(validation_data)}")
        (X_train, y_train) = training_data
        (X_val, y_val) = validation_data
        print_cust(f"client_type: {client_type}, client_idx: {client_idx}, X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")

  client_params_dict = initialize_client_params(clients_config_arg, model_size=min_size_clients, cur_client_params_dict=None, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params, train_models_parallel=train_models_parallel)

  cur_shared_params = None

  client_types = sorted(list(clients_config_arg.keys()))

  data_logs = {}

  data_logs["clients_data_dict"] = clients_data_dict
  data_logs["testing_data"] = (X_test, y_test)

  if save_pkl:
    with open(f"data_logs_n_samples_{n_samples}_dataset_type_{dataset_type}_classes_{'_'.join(classes)}.pkl", "wb") as file:
      pickle.dump(data_logs, file)

  for cur_model_size in client_types:
    print_cust(f"run_qfl_experiments, cur_model_size: {cur_model_size}")
    data_logs[cur_model_size] = {}
    for client_size, cfg in clients_config_arg.items():
      data_logs[cur_model_size][client_size] = []
      # TODO: add a field for testing loss in data_logs
      data_logs[cur_model_size]["aggregated_params_rounds"] = []
      data_logs[cur_model_size]["testing_loss_rounds"] = []
      data_logs[cur_model_size]["testing_accuracy_rounds"] = []
      for client_idx in range(cfg["num_clients"]):
        data_logs[cur_model_size][client_size].append({"trained_params_rounds": [],
                                                       "minibatch_losses_rounds": [],
                                                       "validation_losses_rounds": [],
                                                       "training_accs_rounds": []})


    num_rounds = clients_config_arg[cur_model_size]["communication_rounds"]
    num_local_epochs = clients_config_arg[cur_model_size]["local_epochs"]
    # TODO: probably have some kind of aggregation of metrics here. what do you want to aggregate?

    # TODO: create qnode
    expand = False
    if cur_model_size > min_size_clients:
      expand = True
    feature_list, expansion_data = create_feat_list_expansion_data(cur_model_size, int(math.log2(cur_model_size)), expand=expand, pool_in=pool_in, min_qubits_noexpand=min_size_clients, train_models_parallel=train_models_parallel)

    for round_num in range(num_rounds):
      print_cust(f"run_qfl_experiments, round_num: {round_num}")
      for client_type, client_params in client_params_dict.items():
        print_cust(f"run_qfl_experiments, client_type: {client_type}")
        client_data = clients_data_dict[client_type]
        for client_idx in range(len(client_params)):
          print_cust(f"run_qfl_experiments, client_idx: {client_idx}")
          client_params_indiv = client_params[client_idx]
          client_data_indiv = client_data[client_idx]

          client_train_data = client_data_indiv[0]
          client_val_data = client_data_indiv[1]

          if debug:
            print_cust(f"run_qfl_experiments, len(client_train_data): {len(client_train_data)}")
            print_cust(f"run_qfl_experiments, client_train_data: {client_train_data}")
            print_cust(f"run_qfl_experiments, len(client_val_data): {len(client_val_data)}")
            print_cust(f"run_qfl_experiments, client_val_data: {client_val_data}")

          # train for num_local_epochs

          grad_mask = None

          # TODO: move this conditional up?
          if mask_grads:
            if cur_model_size > min_size_clients:
              grad_mask = mask_gradients(client_params_indiv)

          print_cust(f"run_qfl_experiments, grad_mask: {grad_mask}")

          trained_params, minibatch_losses, validation_losses = train_epochs_angle_param_adam(
        client_params_indiv, client_train_data[0][:, feature_list], client_train_data[1], client_val_data[0][:, feature_list], client_val_data[1],
        n_epochs=num_local_epochs, shots=shots, batch_size=local_batch_size, lr=local_lr, qnode=qnode, trainable_mask=grad_mask)

          client_params[client_idx] = trained_params

          qnode = create_qnode_qcnn(cur_model_size, int(math.log2(cur_model_size)), expansion_data, n_classes=len(classes))

          train_acc = compute_avg_acc_angle_param_batch(trained_params, client_train_data[0][:, feature_list], client_train_data[1], layers=int(np.log2(cur_model_size)), shots=1024, batch_size=local_batch_size, qnode=qnode)

          data_logs[cur_model_size][client_type][client_idx]["training_accs_rounds"].append(train_acc)

          data_logs[cur_model_size][client_type][client_idx]["trained_params_rounds"].append(trained_params)
          data_logs[cur_model_size][client_type][client_idx]["minibatch_losses_rounds"].append(minibatch_losses)
          data_logs[cur_model_size][client_type][client_idx]["validation_losses_rounds"].append(validation_losses)

      # perform aggregation
      # NOTE: assume, during aggregation, that the parameters for all the clients have the same shape.
      print_cust(f"run_qfl_experiments, parameter aggregation, agg_strategy: {agg_strategy}")
      if agg_strategy == "fedavg":
        aggregated_params = federated_averaging(client_params_dict, clients_data_dict)
      elif agg_strategy == "fedavg_quat":
        aggregated_params = federated_averaging_quat(client_params_dict, clients_data_dict)
      elif agg_strategy == "fedavg_circ":
        aggregated_params = federated_averaging_circular(client_params_dict, clients_data_dict)
        # print_cust(f"run_qfl_experiments, aggregated_params: {aggregated_params}")

      data_logs[cur_model_size]["aggregated_params_rounds"].append(aggregated_params)

      # broadcast parameter updates
      for client_type, client_params in client_params_dict.items():
        for client_idx in range(len(client_params)):
          client_params[client_idx] = aggregated_params

      test_loss = compute_loss_angle_param_batch(aggregated_params, X_test[:, feature_list], y_test,
                                            shots=shots, qnode=qnode)

      test_acc = compute_avg_acc_angle_param_batch(aggregated_params, X_test[:, feature_list], y_test, layers=int(np.log2(cur_model_size)), shots=1024, batch_size=local_batch_size, qnode=qnode)

      print_cust(f"run_qfl_experiments, test_loss: {test_loss}")

      print_cust(f"run_qfl_experiments, test_acc: {test_acc}")

      cur_shared_params = aggregated_params

      data_logs[cur_model_size]["testing_loss_rounds"].append(test_loss)

      data_logs[cur_model_size]["testing_accuracy_rounds"].append(test_acc)

    # initialize parameters for the next size of model training
    # TODO: handle different sized models
    print_cust(f"run_qfl_experiments, initializing next-sized models")
    client_params_dict = initialize_client_params(clients_config_arg, model_size=cur_model_size * 2, cur_client_params_dict=client_params_dict, qubits_and_layers_to_add_block_params=qubits_and_layers_to_add_block_params)

  return data_logs