from quorus.logging.custom_slog import print_cust
from sklearn.decomposition import PCA
import torch
from pennylane import numpy as np

"""## Federated PCA Function"""

# TODO, DepthFL: continue here, 8/19 12:12 PM

def perform_federated_pca_mocked(clients_data_dict, max_size_clients, random_seed=42, math_int=np, device=None, generative=False):
  # local_covariances = []
  # local_weights = []
  # pca_list = []
  # for client_type, client_data in clients_data_dict.items():
  #   for client_data_indiv in client_data:
  #     cli_weight = client_data_indiv[0][0].shape[0] + client_data_indiv[1][0].shape[0]
  #     cli_pca = client_data_indiv[2][0]
  #     cli_cov = cli_pca.get_covariance()
  #     local_covariances.append(cli_cov)
  #     local_weights.append(cli_weight)
  #     pca_list.append(cli_pca)
  # total_weights = sum(local_weights)
  # local_weights = [loc_weight / total_weights for loc_weight in local_weights]
  # # assumes that all cov matrices are the same shape
  # assert len(local_covariances) == len(local_weights) == len(pca_list), "Local covariances and local weights length are not the same"
  # global_cov = sum(local_weights[i] * local_covariances[i] for i in range(len(local_covariances)))
  # e_vals, e_vecs = np.linalg.eigh(global_cov)
  # # TODO: again, improve this for the homogenous setting
  # idx = np.argsort(e_vals)[::-1][:max_size_clients]
  # U_global = e_vecs[:, idx]

  # pca_global = PCA(n_components=max_size_clients)
  # pca_global.components_ = U_global.T
  # pca_global.explained_variance_ = e_vals[idx]
  # pca_global.explained_variance_ratio_ = e_vals[idx] / e_vals.sum()
  # pca_global.mean_ = sum(local_weights[i] * pca_list[i].mean_ for i in range(len(pca_list)))
  # pca_global.n_samples_ = sum(pca_list[i].n_samples_ for i in range(len(pca_list)))
  # pca_global.n_features_in_ = U_global.shape[0]

  print_cust(f"perform_federated_pca_mocked, generative: {generative}")

  any_client_data = next(iter(next(iter(clients_data_dict.values()))))

  print_cust(f"perform_federated_pca_mocked, len(any_client_data): {len(any_client_data)}")

  data_dim = any_client_data[0][0].shape[1]

  print_cust(f"perform_federated_pca_mocked, data_dim: {data_dim}")

  agg_client_data = math_int.empty((0, data_dim))

  print_cust(f"perform_federated_pca_mocked, agg_client_data.shape: {agg_client_data.shape}")

  # TODO: this is slow; can change later.
  for client_type, client_data in clients_data_dict.items():
    for client_data_indiv in client_data:
      all_client_data = math_int.concatenate((client_data_indiv[0][0], client_data_indiv[1][0]), axis=0)
      print_cust(f"perform_federated_pca_mocked, all_client_data.shape: {all_client_data.shape}")
      agg_client_data = math_int.concatenate((agg_client_data, all_client_data), axis=0)

  print_cust(f"perform_federated_pca_mocked, max_size_clients: {max_size_clients}")
  # NOTE: PCA is dependent on the seed given.
  pca_global = PCA(n_components=max_size_clients, random_state=random_seed)
  agg_client_data_numpy = agg_client_data.detach().cpu().numpy()
  glob_data_numpy = pca_global.fit_transform(agg_client_data_numpy)
  glob_data = torch.from_numpy(glob_data_numpy).to(device)
  print_cust(f"perform_federated_pca_mocked, glob_data.shape: {glob_data.shape}")
  shared_max_comps_test = math_int.max(glob_data, axis=0)
  shared_min_comps_test = math_int.min(glob_data, axis=0)
  print_cust(f"perform_federated_pca_mocked, shared_max_comps_test: {shared_max_comps_test}, shared_min_comps_test: {shared_min_comps_test}")

  glob_data_invtransform_numpy = pca_global.inverse_transform(glob_data_numpy)
  glob_data_invtransform = torch.from_numpy(glob_data_invtransform_numpy).to(device)

  inv_pca_max = glob_data_invtransform.max()
  inv_pca_min = glob_data_invtransform.min()

  print_cust(f"perform_federated_pca_mocked, inv_pca_max: {inv_pca_max}, inv_pca_min: {inv_pca_min}")

  # transform each clients' data
  shared_max_comps = [-float('inf') for _ in range(max_size_clients)]
  shared_min_comps = [float('inf') for _ in range(max_size_clients)]

  for client_type, client_data in clients_data_dict.items():
    for client_data_indiv in client_data:
      cli_train_data = client_data_indiv[0][0]
      cli_val_data = client_data_indiv[1][0]
      n_cli_train = client_data_indiv[0][0].shape[0]
      cli_train_val_data = math_int.concatenate((cli_train_data, cli_val_data), axis=0)
      cli_train_val_data_numpy = cli_train_val_data.detach().cpu().numpy()
      cli_train_val_data_reduced_numpy = pca_global.transform(cli_train_val_data_numpy)
      cli_train_val_data_reduced = torch.from_numpy(cli_train_val_data_reduced_numpy).to(device)
      assert len(shared_max_comps) <= cli_train_val_data_reduced.shape[1], "Maximum components list doesn't cover all components for client data"
      for i in range(cli_train_val_data_reduced.shape[1]):
          comp = cli_train_val_data_reduced[:, i]
          lo, hi = comp.min(), comp.max()
          if lo < shared_min_comps[i]:
            shared_min_comps[i] = lo
          if hi > shared_max_comps[i]:
            shared_max_comps[i] = hi

      client_data_indiv[2][1] = cli_train_val_data_reduced
      client_data_indiv[0][0] = cli_train_val_data_reduced[:n_cli_train]
      client_data_indiv[1][0] = cli_train_val_data_reduced[n_cli_train:]
      client_data_indiv[2][0] = pca_global

  print_cust(f"perform_federated_pca_mocked, shared_max_comps: {shared_max_comps}, shared_min_comps: {shared_min_comps}")

  if not generative:
    print_cust(f"perform_federated_pca_mocked, rescaling training data to be from 0 to pi")
    for client_type, client_data in clients_data_dict.items():
      for client_data_indiv in client_data:
        cli_train_val_data_reduced = client_data_indiv[2][1]

        # 2) scale each component to [0, π]
        # NOTE, layers: I might get dtype issues here, in which case I'd need to change to torch.float32 explicitly.
        X_angle_cli = math_int.zeros_like(cli_train_val_data_reduced)
        for i in range(cli_train_val_data_reduced.shape[1]):
            comp = cli_train_val_data_reduced[:, i]
            lo, hi = shared_min_comps[i], shared_max_comps[i]
            X_angle_cli[:, i] = ( (comp - lo) / (hi - lo + 1e-8) ) * math_int.pi

        # ADDED, layers: n_cli_train, as it should be specified for this particular client_data_indiv.
        n_cli_train = client_data_indiv[0][0].shape[0]

        cli_train_data_reduced = X_angle_cli[:n_cli_train]
        cli_val_data_reduced = X_angle_cli[n_cli_train:]

        client_data_indiv[0][0] = cli_train_data_reduced
        client_data_indiv[1][0] = cli_val_data_reduced

        # client_data_indiv[2][0] = pca_global
        # client_data_indiv[2][1] = cli_train_val_data_reduced
        # NOTE, layers: shouldn't need to change [2][1]; i.e., the PCA reduced but non-angle encoded data right?

  return pca_global, shared_min_comps_test, shared_max_comps_test, inv_pca_min, inv_pca_max