from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
import math
from quorus.data_ops.data_processing import angle_encode_data

"""## Split data federated function"""

def split_data_federated(X, y, client_config, test_frac, val_frac=0.2,
                         feature_skew=0.0, label_skew=None, random_state=42, local_pca=False, do_lda=False, feat_sel_type="top", amp_embed=False, feat_ordering="same",
                         shared_pca=False, fed_pca_mocked=False):
    """
    Splits data for federated learning, inducing controllable label‐skew
    that goes from uniform (label_skew=1) to linear (s=0.5) to extreme (s→0).

    Parameters:
      X: The input data to split among clients.
      y: The input labels to split among clients.
      client_config: The client configuration dictionary used to determine how much data each client of each type gets.
      test_frac: Float representing the fraction of the input data used for testing.
      val_frac: Float representing the fraction of each clients' data used for validation.
      feature_skew: Float representing the strength of the skew that each client faces in terms of feature values.
      label_skew: None or float in [0,1].
        - 1.0 => each client’s labels are uniform.
        - 0.5 => exactly linear descending (green line).
        - 0.0 => extreme “red” skew (almost all mass on class 0).
        - Values in between smoothly interpolate via exponent θ = (1−s)/s.
      random_state: Integer representing the random state used for the entire program (TODO: change this; this function now changes global state)
      local_pca: Boolean indicating whether or not PCA should be performed locally.
      do_lda: Boolean indicating whether or not random sketching should be performed.
      feat_sel_type: String representing the choice of features that the client will take later.
      amp_embed: Boolean indicating whether or not the data will be amplitude encoded.
      feat_ordering: String representing the ordering of features to sample in the case where data is amplitude encoded (TODO: use the same logic for sampling amplitude encoded or
      angle encoded data, or refactor so that this is true)

    Returns:
      clients_data: a dictionary mapping integers representing client types to a list of data for each client, where the i-th element of the list is the data for
      client i, and each client has a list of data in the form [(X_train, y_train), (X_val, y_val), (pca_obj, pca_reduced_data)]
      with the third element only being present if local PCA is performed
      (X_test, y_test): a tuple of testing data and labels

    """
    print_cust(f"split_data_federated, fed_pca_mocked: {fed_pca_mocked}")
    print_cust(f"split_data_federated, client_config: {client_config}")
    # TODO: test this function. i keep getting weird results where the acc for each client differs a lot ...

    X = np.array(X)
    y = np.array(y)
    if random_state is not None:
        np.random.seed(random_state)

    # 1) global train/test split
    n = len(X)
    perm = np.random.permutation(n)
    tsize = int(test_frac * n)
    test_idx, train_idx = perm[:tsize], perm[tsize:]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    n_train = len(X_train)

    rel_idx = np.arange(n_train)
    pointer = 0
    clients_data = {}

    max_cli_size = max(client_config.keys())

    if do_lda:
      sketch_mat = np.random.normal(loc=0.0, scale=1/np.sqrt(X.shape[1]), size=(X.shape[1], max_cli_size))

    # sanity
    total_pct = sum(cfg["percentage_data"] for cfg in client_config.values())
    if total_pct > 1.0:
        raise ValueError("Sum of percentage_data > 1")

    # 2) per-client-type allocation
    for ctype, cfg in client_config.items():
        pct, n_clients = cfg["percentage_data"], cfg["num_clients"]

        # carve off this type’s pool
        alloc_n = int(pct * n_train)
        alloc_n = min(alloc_n, n_train - pointer)
        alloc_idx = rel_idx[pointer : pointer + alloc_n]
        pointer += alloc_n

        # --- feature skew (unchanged) ---
        feats = X_train[alloc_idx, 0].astype(float)
        if feats.size:
            lo, hi = feats.min(), feats.max()
            norm_feat = (feats - lo)/(hi - lo) if hi > lo else np.zeros_like(feats)
        else:
            norm_feat = feats
        rand_comp = np.random.rand(len(alloc_idx))
        scores = feature_skew*norm_feat + (1-feature_skew)*rand_comp
        alloc_idx = alloc_idx[np.argsort(scores)]

        # --- label skew ---
        # --- label skew (robust, disjoint sampling) ---
        if label_skew is None:
            client_chunks = np.array_split(alloc_idx, n_clients)
        else:
            labels_sorted = np.array(sorted(np.unique(y_train[alloc_idx])))
            C             = len(labels_sorted)

            # build the rank‐power weights w_i = (C−i)^θ
            s     = float(label_skew)
            theta = (1.0 - s)/s if s>0 else np.inf
            ranks = np.arange(C, 0, -1, dtype=float)    # [C, C-1, ...,1]
            w     = (ranks**theta) if np.isfinite(theta) else np.zeros_like(ranks)
            if not np.isfinite(theta):
                w[0] = 1.0   # all mass on first class
            p_local = w / w.sum()

            # we’ll sample from this array, removing as we go
            remaining = np.array(alloc_idx, dtype=int)
            # precompute a map from label → slot in p_local
            lbl2idx = {lbl:i for i,lbl in enumerate(labels_sorted)}

            client_chunks = []
            for client_id in range(n_clients):
                n_rem        = len(remaining)
                if n_rem == 0:
                    client_chunks.append(np.array([],dtype=int))
                    continue

                clients_left = n_clients - client_id
                take_n       = int(math.ceil(n_rem / clients_left))

                # build per-sample probs
                classes_rem = y_train[remaining]
                q           = np.array([p_local[lbl2idx[l]] for l in classes_rem])
                q          /= q.sum()

                # sample without replacement
                sel_idx = np.random.choice(
                    n_rem,
                    size=min(take_n, n_rem),
                    replace=False,
                    p=q
                )
                sel      = remaining[sel_idx]
                client_chunks.append(sel)

                # remove them
                remaining = np.delete(remaining, sel_idx)

        # 3) split each client chunk into train/val
        clients_data[ctype] = []
        for chunk in client_chunks:
            m = len(chunk)
            v = int(val_frac * m)
            client_data_chunk = X_train[chunk]
            val_idx   = chunk[:v]
            train_idx = chunk[v:]
            n_tot_comps = ctype
            # Have ALL components so that we have the option to sample others (as opposed to just the top components).
            if feat_sel_type != "top" or shared_pca:
              n_tot_comps = max_cli_size
            # Perform local PCA (and local random sketching) if specified.
            if local_pca and not fed_pca_mocked:
              if do_lda:
                # TODO: validation data prob should not be LDA'd together w/ training, but it's alright for now.. doesn't affect training for single client epoch case
                client_data_chunk, pca, client_data_chunk_pca = angle_encode_data(client_data_chunk, n_tot_comps, y=y_train[chunk], do_lda=True, ret_lda=True, sketch_mat=sketch_mat[:, :ctype])
              else:
                client_data_chunk, pca, client_data_chunk_pca = angle_encode_data(client_data_chunk, n_tot_comps, do_pca=True, ret_pca=True)
            else:
              client_data_chunk, pca, client_data_chunk_pca = client_data_chunk, None, None

            # If the data will be amplitude encoded, and the features are sampled in order of highest variance,
            # add a small constant to the smallest valued pixel so that the data can be subsequently normalized.
            if amp_embed and feat_ordering == "highest_var":
              variances = X.var(axis=0, ddof=0)
              order = np.argsort(variances)[::-1]
              client_data_chunk = client_data_chunk[:, order] + 1e-3

            # Store the original data for this particular client.
            if shared_pca:
              client_data_chunk = X_train[chunk]

            X_train_data = client_data_chunk[v:]
            X_val_data = client_data_chunk[:v]
            client_data_lst = [[X_train_data, y_train[train_idx]],
                 [X_val_data,   y_train[val_idx]]]
            if local_pca:
              client_data_lst.append([pca, client_data_chunk_pca])
            clients_data[ctype].append(
                client_data_lst
            )

    return clients_data, (X_test, y_test)