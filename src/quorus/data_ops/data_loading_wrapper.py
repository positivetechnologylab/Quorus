from quorus.logging.custom_slog import print_cust
from quorus.data_ops.data_loading import load_mnist_digits, load_cifar10
import pickle
from quorus.data_ops.data_processing import angle_encode_data
from sklearn.datasets import make_classification
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pennylane import numpy as np

def load_dataset(dataset_type="mnist", classes=["4", "9"], n_samples=1000, num_feats=20, keep_orig_imgs=False, do_lda=False, custom_debug=False):
    """
    Load a dataset and return it as angles together with labels.

    Parameters:
      dataset_type (str): Can be:
           "mnist"      -- load MNIST digits (using classes list, e.g. ["4", "9"])
           "synthetic"  -- generate synthetic binary classification data
           "pima"       -- load the Pima Indians Diabetes dataset from OpenML.
           "higgs"      -- load the HIGGS dataset from OpenML.
           "covertype"  -- load the Covertype dataset from OpenML (binarized).
      classes (list): Used only when dataset_type is "mnist". (Ignored for other types.)
      n_samples (int): Number of samples to load or generate.
      num_feats (int): Number of features (or PCA components) to extract and encode as angles.
      keep_orig_imgs (Boolean): Boolean indicating whether or not the original images should be kept (and no dimensionality reduction should be done)
      do_lda (Boolean): Boolean indicating whether or not random sketching should be performed.

    Returns:
      X_angles (np.array): Array of shape (n_samples, num_feats) with values scaled to [0, π].
      y (np.array): Binary labels.
    """
    do_pca = False
    if not do_lda:
      do_pca = True

    if custom_debug:
      print_cust(f"load_dataset, do_pca: {do_pca}, do_lda: {do_lda}")

    if dataset_type == "mnist" or dataset_type == "Fashion-MNIST":
        print_cust(f"load_dataset, is {dataset_type}")
        if dataset_type == "mnist":
          dataset_type = "mnist_784"
        if custom_debug:
          print_cust(f"load_dataset, dataset_type: {dataset_type}")
        # Assumes a function load_mnist_digits exists.
        X, y = load_mnist_digits(classes, n_samples=n_samples, dataset_name=dataset_type)
        if custom_debug:
          # with open('load_mnist_digits_X.txt', 'w') as f:
          #   print_cust(X, file=f)
          # with open('load_minst_digits_y.txt', 'w') as f:
          #   print_cust(y, file=f)
          with open('load_mnist_digits_X.pkl', 'wb') as f:
            pickle.dump(X, f)
          with open('load_mnist_digits_y.pkl', 'wb') as f:
            pickle.dump(y, f)
        if keep_orig_imgs:
          X_angles = X
        else:
          X_angles = angle_encode_data(X, y=y, n_components=num_feats, do_pca=do_pca, do_lda=do_lda, custom_debug=custom_debug)
        return X_angles, y.astype(int)

    elif dataset_type == "cifar10":
        print_cust("load_dataset, is cifar10")
        X, y = load_cifar10(classes, n_samples=n_samples)
        # CIFAR-10 images are higher-dimensional; applying PCA is recommended.
        X_angles = angle_encode_data(X, y=y, n_components=num_feats, do_pca=do_pca, do_lda=do_lda)
        return X_angles, y.astype(int)

    elif dataset_type == "synthetic":
        print_cust("load_dataset, is synthetic")
        # Generate synthetic binary classification data.
        # NOTE: func is not defined
        X, y = make_classification(n_samples=n_samples,
                                   n_features=num_feats,
                                   n_informative=num_feats,
                                   n_redundant=0,
                                   n_repeated=0,
                                   n_classes=2,
                                   random_state=42)
        y = y.astype(int)
        X_angles = angle_encode_data(X, y=y, n_components=num_feats, do_pca=do_pca, do_lda=do_lda)
        return X_angles, y

    elif dataset_type == "pima":
        print_cust("load_dataset, is pima")
        # Load the Pima Indians Diabetes dataset from OpenML.
        pima = fetch_openml('diabetes', version=1, as_frame=False)
        X = pima.data
        print_cust(f"load_dataset, X.shape: {X.shape}")
        mapping = {"tested_negative": 0, "tested_positive": 1}
        y = np.array([mapping.get(label, 0) for label in pima.target])
        if n_samples < len(y):
            X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)
        X_angles = angle_encode_data(X, y=y, n_components=num_feats, do_pca=do_pca, do_lda=do_lda)
        return X_angles, y

    elif dataset_type == "higgs":
        print_cust("load_dataset, is higgs")
        # Load the HIGGS dataset from OpenML.
        higgs = fetch_openml('HIGGS', version=1, as_frame=False)
        X = higgs.data
        # The target is already numeric; convert to int if necessary.
        y = higgs.target.astype(int)
        if n_samples < len(y):
            X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)
        X_angles = angle_encode_data(X, y=y, n_components=num_feats, do_pca=do_pca, do_lda=do_lda)
        return X_angles, y

    elif dataset_type == "covertype":
        print_cust("load_dataset, is covertype")
        # Load the Covertype dataset from OpenML.
        covertype = fetch_openml('covertype', version=1, as_frame=False)
        X = covertype.data
        # Convert the string labels to integers using LabelEncoder.
        le = LabelEncoder()
        y_encoded = le.fit_transform(covertype.target)
        print_cust("Covertype classes:", le.classes_)
        # Binarize: designate the label that is encoded as 1 as the positive class.
        y_binary = (y_encoded == 1).astype(int)
        if n_samples < len(y_binary):
            X, _, y_binary, _ = train_test_split(X, y_binary, train_size=n_samples, stratify=y_binary, random_state=42)
        X_angles = angle_encode_data(X, y=y_binary, n_components=num_feats, do_pca=do_pca, do_lda=do_lda)
        return X_angles, y_binary

    elif dataset_type == "breast_cancer":
        # if custom_debug:
        print_cust("load_dataset, is breast_cancer")

        # scikit-learn ships the 569×30 WDBC matrix locally:contentReference[oaicite:1]{index=1}
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        print_cust(f"load_dataset, X.shape: {X.shape}")
        print_cust(f"load_dataset, np.unique(y): {np.unique(y)}")

        # Optional down-sampling if the caller asked for fewer rows
        if n_samples < len(y):
            print_cust(f"load_dataset, n_samples < len(y)")

            X, _, y, _ = train_test_split(X, y,
                                          train_size=n_samples,
                                          stratify=y,
                                          random_state=42)

            print_cust(f"load_dataset, X.shape: {X.shape}, y.shape: {y.shape}")

        # Angle-encoding: either raw 30-D or reduced to `num_feats`
        if keep_orig_imgs:
            # keep_orig_imgs means “no reduction”; we pass X straight through
            print_cust(f"load_dataset, keep_orig_imgs, so passing X_angles = X")
            X_angles = X
        else:
            X_angles = angle_encode_data(X, y=y,
                                         n_components=num_feats,
                                         do_pca=do_pca,
                                         do_lda=do_lda,
                                         custom_debug=custom_debug)
        print_cust(f"load_dataset, X_angles: {X_angles}, y: {y}, y.astype(int): {y.astype(int)}")
        return X_angles, y.astype(int)

    # ------------------------------------------------------------
    # 4) Unknown keyword guard
    # ------------------------------------------------------------
    else:
        raise ValueError(
            "Unknown dataset_type. "
            "Choose from 'mnist', 'cifar10', 'synthetic', 'pima', "
            "'higgs', 'covertype', or 'breast_cancer'."
        )