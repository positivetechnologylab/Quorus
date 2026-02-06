from quorus.logging.custom_slog import print_cust
from pennylane import numpy as np
import math
from quorus.data_ops.data_processing import angle_encode_data

"""## Split data federated function"""

def split_data_federated(
    X, y, client_config, test_frac, val_frac=0.2,
    feature_skew=0.0, label_skew=None, random_state=42,
    local_pca=False, do_lda=False, feat_sel_type="top",
    amp_embed=False, feat_ordering="same",
    shared_pca=False, fed_pca_mocked=False
):
    """
    Splits data for federated learning.

    New semantics:
      - cfg["percentage_data"] is interpreted as *per-client* fraction
        of the training data (after the global train/test split).
      - Each individual client j gets:
            n_j = int(cfg["percentage_data"] * n_train) samples.
      - If label_skew is not None, the split is non-IID in labels using
        a Dirichlet-based label distribution as in DepthFL / FedMA:
            For each class c:
                p_c ~ Dir_K(beta = label_skew)
                allocate class-c samples to the K clients according to p_c
        subject to each client's capacity n_j.

    Parameters:
      X, y: full dataset.
      client_config: dict[client_type] -> {
          "percentage_data": float in [0,1]  (per-client share),
          "num_clients": int
      }
      test_frac: fraction of full data used as a global test set.
      val_frac: fraction of *each client’s* data used as validation.
      feature_skew: (kept, but only meaningfully used when label_skew is None).
      label_skew: None or float in [0,1].
        - None: near-IID label distribution (random sampling).
        - s in (0,1]: Dirichlet label skew with concentration beta = s
          (smaller s => stronger label non-IID).
      random_state, local_pca, do_lda, feat_sel_type, amp_embed,
      feat_ordering, shared_pca, fed_pca_mocked: as in your original code.

    Returns:
      clients_data: dict[ctype] -> list over that type’s clients of:
          [[X_train_i, y_train_i], [X_val_i, y_val_i]]
          (+ [pca_obj, pca_reduced_data] if local_pca)
      (X_test, y_test): global test split.
    """
    print_cust(f"split_data_federated, fed_pca_mocked: {fed_pca_mocked}")
    print_cust(f"split_data_federated, client_config: {client_config}")

    X = np.array(X)
    y = np.array(y)
    if random_state is not None:
        np.random.seed(random_state)

    # 1) global train/test split
    n = len(X)
    perm_all = np.random.permutation(n)
    tsize = int(test_frac * n)
    test_idx, train_idx = perm_all[:tsize], perm_all[tsize:]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    n_train = len(X_train)

    max_cli_size = max(client_config.keys())

    if do_lda:
        sketch_mat = np.random.normal(
            loc=0.0,
            scale=1 / np.sqrt(X.shape[1]),
            size=(X.shape[1], max_cli_size)
        )

    # ---- Build a flat list of all clients (across all types) ----
    # Each entry: {"ctype": ctype, "local_id": i, "pct": percentage_data}
    client_list = []
    for ctype, cfg in client_config.items():
        pct = cfg["percentage_data"]
        n_clients = cfg["num_clients"]
        for i in range(n_clients):
            client_list.append({"ctype": ctype, "local_id": i, "pct": pct})

    K = len(client_list)
    if K == 0:
        raise ValueError("client_config specifies zero clients.")

    # sanity: sum over *clients* of percentage_data must be <= 1
    total_pct_clients = sum(cli["pct"] for cli in client_list)
    if total_pct_clients > 1.0 + 1e-8:
        raise ValueError(
            f"Sum over clients of percentage_data ({total_pct_clients:.4f}) > 1.0. "
            "Ensure sum(cfg['percentage_data'] * num_clients) <= 1.0."
        )

    # Per-client capacities (training samples each should receive)
    capacities = np.array(
        [int(cli["pct"] * n_train) for cli in client_list],
        dtype=int
    )
    total_cap = capacities.sum()
    if total_cap == 0:
        raise ValueError("All clients have zero capacity (percentage_data too small?).")

    if total_cap > n_train:
        # Strictly speaking this should not happen if total_pct_clients <= 1,
        # but we guard in case of rounding issues.
        raise RuntimeError(
            f"Total requested samples ({total_cap}) exceed available train data ({n_train})."
        )

    # 2) Build client -> list of assigned training indices
    client_indices = [[] for _ in range(K)]

    if label_skew is None:
        # ---- Near-IID case: just random sampling, respecting capacities ----
        all_idx = np.random.permutation(n_train)
        cursor = 0
        for j, cap in enumerate(capacities):
            if cap <= 0:
                continue
            end = cursor + cap
            if end > n_train:
                end = n_train
            client_indices[j].extend(all_idx[cursor:end].tolist())
            cursor = end
            if cursor >= n_train:
                break
        # leftover training samples (if any) remain unassigned
    else:
        # ---- DepthFL-style non-IID: Dirichlet over labels, with capacity caps ----
        s = float(label_skew)
        if not (0.0 <= s <= 1.0):
            raise ValueError("label_skew must be None or a float in [0, 1].")
        alpha = max(s, 1e-3)  # Dirichlet concentration

        remaining_cap = capacities.copy()

        classes = np.unique(y_train)
        # Precompute & shuffle indices per class
        class_indices = {}
        for c in classes:
            idx_c = np.where(y_train == c)[0]
            idx_c = np.random.permutation(idx_c)
            class_indices[c] = idx_c

        # For each class, draw p_c ~ Dir_K(alpha) and assign its samples
        for c in classes:
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            # Dirichlet over K clients for this class
            p_c = np.random.dirichlet(alpha * np.ones(K))

            for idx in idx_c:
                # Clients that still have remaining capacity
                available = np.where(remaining_cap > 0)[0]
                if available.size == 0:
                    break  # all clients filled their quotas

                p_avail = p_c[available]
                if p_avail.sum() <= 0:
                    p_avail = np.ones_like(p_avail) / len(p_avail)
                else:
                    p_avail = p_avail / p_avail.sum()

                chosen_local = np.random.choice(len(available), p=p_avail)
                j = available[chosen_local]

                client_indices[j].append(idx)
                remaining_cap[j] -= 1

                if remaining_cap.sum() == 0:
                    break  # all capacities used

            if remaining_cap.sum() == 0:
                break  # all capacities used

        # At this point, each client j should have <= capacities[j] samples.
        # In practice, we expect equality for all clients with cap > 0.
        for j, cap in enumerate(capacities):
            if cap > 0 and len(client_indices[j]) != cap:
                print_cust(
                    f"[split_data_federated] WARNING: client {j} "
                    f"has {len(client_indices[j])} samples, expected {cap}."
                )

    # Optionally apply feature_skew in the IID-like case by reordering indices
    if label_skew is None and feature_skew > 0.0:
        # Simple version: reorder within each client's chunk based on first feature
        for j in range(K):
            idxs = np.array(client_indices[j], dtype=int)
            if idxs.size == 0:
                continue
            feats = X_train[idxs, 0].astype(float)
            lo, hi = feats.min(), feats.max()
            if hi > lo:
                norm_feat = (feats - lo) / (hi - lo)
            else:
                norm_feat = np.zeros_like(feats)
            rand_comp = np.random.rand(len(idxs))
            scores = feature_skew * norm_feat + (1 - feature_skew) * rand_comp
            client_indices[j] = idxs[np.argsort(scores)].tolist()

    # 3) Group indices back by client type and build per-client data
    # indices_by_type[ctype][local_id] = np.array([...])
    indices_by_type = {
        ctype: [np.array([], dtype=int) for _ in range(cfg["num_clients"])]
        for ctype, cfg in client_config.items()
    }

    for j, cli in enumerate(client_list):
        ctype = cli["ctype"]
        local_id = cli["local_id"]
        idxs = np.array(client_indices[j], dtype=int)
        indices_by_type[ctype][local_id] = idxs

    clients_data = {}
    for ctype, cfg in client_config.items():
        n_clients_type = cfg["num_clients"]
        clients_data[ctype] = []
        for client_idx in range(n_clients_type):
            chunk = indices_by_type[ctype][client_idx]
            m = len(chunk)
            if m == 0:
                # This client gets no data
                X_train_data = np.empty((0,) + X_train.shape[1:], dtype=X_train.dtype)
                y_train_data = np.empty((0,), dtype=y_train.dtype)
                X_val_data = X_train_data
                y_val_data = y_train_data
                client_data_lst = [
                    [X_train_data, y_train_data],
                    [X_val_data,   y_val_data],
                ]
                clients_data[ctype].append(client_data_lst)
                continue

            v = int(val_frac * m)
            client_data_chunk = X_train[chunk]
            val_idx   = chunk[:v]
            train_idx = chunk[v:]
            n_tot_comps = ctype

            # If we want to be able to sample non-top components later,
            # use the maximum size across client types.
            if feat_sel_type != "top" or shared_pca:
                n_tot_comps = max_cli_size

            # Local PCA / LDA if specified
            if local_pca and not fed_pca_mocked:
                if do_lda:
                    client_data_chunk, pca, client_data_chunk_pca = angle_encode_data(
                        client_data_chunk,
                        n_tot_comps,
                        y=y_train[chunk],
                        do_lda=True,
                        ret_lda=True,
                        sketch_mat=sketch_mat[:, :ctype]
                    )
                else:
                    client_data_chunk, pca, client_data_chunk_pca = angle_encode_data(
                        client_data_chunk,
                        n_tot_comps,
                        do_pca=True,
                        ret_pca=True
                    )
            else:
                client_data_chunk, pca, client_data_chunk_pca = client_data_chunk, None, None

            # Amplitude embedding feature-order adjustment
            if amp_embed and feat_ordering == "highest_var":
                variances = X.var(axis=0, ddof=0)
                order = np.argsort(variances)[::-1]
                client_data_chunk = client_data_chunk[:, order] + 1e-3

            if shared_pca:
                client_data_chunk = X_train[chunk]

            X_train_data = client_data_chunk[v:]
            X_val_data   = client_data_chunk[:v]
            y_train_data = y_train[train_idx]
            y_val_data   = y_train[val_idx]

            client_data_lst = [
                [X_train_data, y_train_data],
                [X_val_data,   y_val_data],
            ]
            if local_pca:
                client_data_lst.append([pca, client_data_chunk_pca])

            clients_data[ctype].append(client_data_lst)

    return clients_data, (X_test, y_test)