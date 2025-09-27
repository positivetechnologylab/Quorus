from quorus.logging.custom_slog import print_cust
from quorus.quantum_circuit_funcs.utils.qcnn_code import compute_reduced_qubits

"""## Feature List Expansion Data Function"""

def create_feat_list_expansion_data(n_qubits, conv_layers, expand=False, pool_in=False, min_qubits_noexpand=4, train_models_parallel=False, feat_sel_type="top"):
    """
    Create the feature list (ordering of features on qubits) and expansion data (a list of (encoded_angles, reversed_indices) for
    use in identity initialization) for the specified model size and convolutional layer count.

    Parameters:
      n_qubits: Integer representing the number of qubits in the model.
      conv_layers: Integer representing the number of convolutional layers in the model.
      expand: Boolean representing whether or not this model is an expansion of a smaller model.
      pool_in: Boolean indicating whether or not the qubits should be measured/compressed inwards.
      min_qubits_noexpand: Integer representing the minimum number of qubits for which no expansion shoudl take place, across all input models.
      train_models_parallel: Boolean indicating whether or not the models are being trained in parallel (in this case, no identity initialization takes place).
      feat_sel_type: String representing the order in which the features should be chosen.

    Returns:
      feature_list: List of integers representing the ordering of the features on the qubits
      expansion_data: List of (encoded_angle_indices, reversed_indices) for specifying what input data should be initialized to the identity.
    """
    # Store the current feature list, the total number of features, as well as the remaining features.
    cur_feature_list = []
    expansion_data = []
    total_num_feats = compute_reduced_qubits(n_qubits, 0)
    remaining_feats_list = list(range(total_num_feats))

    # num_feats = n_qubits

    # TODO: check/do this only for pool_in
    # For each layer in the convolutional layer, starting from the smallest layer and increasing,
    # compute the feature list and "expand" the layers.
    for layer in range(conv_layers, -1, -1):
        num_feats = compute_reduced_qubits(n_qubits, layer)
        # No expansion data if the number of qubits is less than the minimum number of qubits for expansion.
        if num_feats < min_qubits_noexpand:
          expansion_data.append([])
          continue
        # If this is the "base" number of qubits for expansion, then initialize the feature list and expansion data.
        if num_feats == min_qubits_noexpand:
          # Either take the top features, or the top and lowest features.
          if feat_sel_type == "top":
            cur_feature_list = list(range(num_feats))
          elif feat_sel_type == "toplow":
            half_feats = num_feats // 2
            # cur_feature_list.extend(all_feats[:half_feats])
            # cur_feature_list.extend(all_feats[-half_feats:])
            cur_feature_list = remaining_feats_list[:half_feats] + remaining_feats_list[-half_feats:]
            remaining_feats_list = sorted(set(remaining_feats_list) - set(cur_feature_list))
          expansion_data.append([])
        else:
          # If this is a convolutional layer that is an expansion of a smaller layer, then for the feature list,
          # insert features to minimize the distance of the remaining qubits that are not pooled.
          # Store the order of features and qubits that were added in the expansion data.
          print_cust(f"experiment_dynamic_QCNN, cur_feature_list: {cur_feature_list}")
          insertion_indices = []
          cur_half_feats = len(cur_feature_list) // 2
          for cur_feat_idx in range(len(cur_feature_list)):
            if cur_feat_idx < cur_half_feats:
              insertion_indices.append(cur_feat_idx * 2)
            else:
              insertion_indices.append(cur_feat_idx * 2 + 1)
          print_cust(f"experiment_dynamic_QCNN, insertion_indices: {insertion_indices}")
          # Either take the top, or the top and lowest features.
          if feat_sel_type == "top":
            new_feat_idxs = list(range(len(cur_feature_list), num_feats))
          elif feat_sel_type == "toplow":
            # all_feat_idxs = sorted(set(range(num_feats)) - set(cur_feature_list))
            num_feats_to_select = num_feats - len(cur_feature_list)
            half_feats_to_select = num_feats_to_select // 2
            new_feat_idxs = remaining_feats_list[:half_feats_to_select] + remaining_feats_list[-half_feats_to_select:]
            remaining_feats_list = sorted(set(remaining_feats_list) - set(new_feat_idxs))
          print_cust(f"experiment_dynamic_QCNN, new_feat_idxs: {new_feat_idxs}")
          for new_feat_idx_idx in range(len(new_feat_idxs)):
            new_feat = new_feat_idxs[new_feat_idx_idx]
            insertion_idx = insertion_indices[new_feat_idx_idx]
            print_cust(f"experiment_dynamic_QCNN, new_feat: {new_feat}")
            print_cust(f"experiment_dynamic_QCNN, insertion_idx: {insertion_idx}")
            cur_feature_list.insert(insertion_idx, new_feat)
            print_cust(f"experiment_dynamic_QCNN, cur_feature_list: {cur_feature_list}")

          # Only add expansion data if we are expanding and want to have an identity iniialization.
          if expand == True:
            expansion_data.append([list(cur_feature_list), insertion_indices])
          else:
            expansion_data.append([])

    feature_list = list(cur_feature_list)

    # If models are trained in parallel, get rid of the expansion data because we do not have/want an identity initialization.
    if train_models_parallel == True:
      new_expansion_data = []
      for layer in range(conv_layers, -1, -1):
        new_expansion_data.append([])
      expansion_data = new_expansion_data

    expansion_data.reverse()

    print_cust(f"create_feat_list_expansion_data, feature_list: {feature_list}")
    print_cust(f"create_feat_list_expansion_data, expansion_data: {expansion_data}")

    return feature_list, expansion_data