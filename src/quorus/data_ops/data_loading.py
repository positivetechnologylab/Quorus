"""# Data Loading

## Load MNIST Digits Helpers
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from urllib.error import HTTPError, URLError

from pennylane import numpy as np

from quorus.logging.custom_slog import print_cust

# TOMODIFY, layers: these are globals and are ONLY here to support names for fashion. it's not exactly necessary.
_FASHION_ID_TO_NAME = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}
_FASHION_NAME_TO_ID = {v.lower(): k for k, v in _FASHION_ID_TO_NAME.items()}

def _normalize_and_flatten(X):
    X = X.astype(np.float32) / 255.0
    if X.ndim == 3:  # (N, 28, 28)
        X = X.reshape((X.shape[0], -1))
    return X

def _canon_label_list(digits_to_keep, is_fashion):
    ids = []
    for d in digits_to_keep:
        s = str(d).strip()
        if s.isdigit():
            ids.append(int(s))
        elif is_fashion and s.lower() in _FASHION_NAME_TO_ID:
            ids.append(_FASHION_NAME_TO_ID[s.lower()])
        else:
            raise ValueError(
                f"Unrecognized class '{d}'. "
                + ("Use 0–9 or Fashion-MNIST names like 'sneaker', 'bag'."
                   if is_fashion else "Use digits 0–9.")
            )
    # dedupe but keep stable order by sorting later
    return sorted(set(ids))

"""## Load MNIST Digits Function"""

def load_mnist_digits(digits_to_keep, n_samples=2000, dataset_name="mnist"):
    """
    Load MNIST or Fashion-MNIST, filter to selected classes, and return (X, y).
    Tries OpenML first; if that errors, falls back to TensorFlow's keras.datasets.

    Parameters:
      digits_to_keep: list of class identifiers. For MNIST: digits (e.g., [4, 9]).
                      For Fashion-MNIST: digits 0–9 or names (e.g., ["sneaker", "bag"]).
      n_samples: number of examples to return (stratified). If >= available, returns all.
      dataset_name: "mnist" or "fashion-mnist" (case/underscore/dash tolerant).

    Returns:
      X: float32 array of shape (N, 784), normalized to [0, 1].
      y: int array of shape (N,), labels remapped to 0..K-1 following sorted(digits_to_keep).
    """
    ds = dataset_name.replace("_", "-").lower()
    # NOTE, layers: can later add some additional conditions to allow for CIFAR-10.
    if ds in {"mnist", "mnist-784"}:
        openml_name = "mnist_784"          # OpenML dataset name
        is_fashion = False
        tf_loader_path = ("tensorflow.keras.datasets.mnist", "mnist")
    elif ds in {"fashion-mnist", "fashion mnist", "fashion"}:
        openml_name = "Fashion-MNIST"      # OpenML dataset name
        is_fashion = True
        tf_loader_path = ("tensorflow.keras.datasets.fashion_mnist", "fashion_mnist")
    else:
        raise ValueError("dataset_name must be 'mnist' or 'fashion-mnist'.")

    keep_ids = _canon_label_list(digits_to_keep, is_fashion)

    # 1) Try OpenML first
    X, y = None, None
    try:
        mnist = fetch_openml(openml_name, version=1, as_frame=False)
        X = _normalize_and_flatten(mnist.data)
        # y can be strings; coerce to int if possible, else map names for fashion
        try:
            y = mnist.target.astype(int)
        except ValueError:
            # NOTE: this should be unnecessary; Fashion-MNIST from OpenML has labels that are string representations of digits.
            if is_fashion:
                y = np.array([_FASHION_NAME_TO_ID[str(lbl).lower()] for lbl in mnist.target])
            else:
                # MNIST should be numeric; re-raise if it's not
                raise
    except (HTTPError, URLError, OSError, RuntimeError, ValueError) as _:
        # 2) Fallback to TensorFlow's Keras loader
        try:
            print_cust(f"load_mnist_digits, USING TENSORFLOW TO LOAD IN MNIST DATA (b/c fetch_openml has errors)")
            print_cust(f"load_mnist_digits, error from fetch_openml code, _: {_}")
            # lazy import to avoid TF dependency unless needed
            import importlib
            mod_name, attr = tf_loader_path
            mod = importlib.import_module(mod_name)
            (X_train, y_train), (X_test, y_test) = getattr(mod, "load_data")()
            X = _normalize_and_flatten(np.concatenate([X_train, X_test], axis=0))
            y = np.concatenate([y_train, y_test], axis=0).astype(int)
        except Exception as e:
            raise RuntimeError(
                f"Both OpenML and TensorFlow loading failed: {type(e).__name__}: {e}"
            )

    # Filter to desired classes
    mask = np.isin(y, keep_ids)
    X, y = X[mask], y[mask]

    # Remap labels to 0..K-1 in sorted order of requested classes
    mapping = {label: idx for idx, label in enumerate(keep_ids)}
    y = np.array([mapping[int(lbl)] for lbl in y], dtype=int)

    # Optional stratified subsample
    if n_samples < len(y):
        # NOTE: because random_state is hardcoded in here, then we randomly select the SAME subset of data in total which is randomly sampled from
        # to create the clients' individual data.
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=42
        )

    return X, y

"""## Load CIFAR-10 Data"""

# from tensorflow.keras.datasets import cifar10

def load_cifar10(classes, n_samples=2000):
    """
    Load the CIFAR-10 dataset, filter by the specified classes,
    and return the filtered data and labels.

    Parameters:
      classes (list): List of classes to keep (either as numeric strings like ["3", "5"]
                      or as names like ["cat", "dog"]).
      n_samples (int): Number of samples to return.

    Returns:
      X (np.array): Normalized and flattened image data.
      y (np.array): Integer labels remapped according to sorted(selected_classes).
    """

    # Load the data from Keras (train and test combined)
    # NOTE: cifar10 is currently not supported because it requires tensorflow.
    # You can write your own implementation for cifar10, so long as it has a load_data() method
    # that returns the data of interest.
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).flatten()

    # CIFAR-10 standard class names mapping
    cifar10_classes = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }

    # Determine if classes are specified as digit strings or names.
    try:
        if classes and classes[0].isdigit():
            selected_indices = [int(c) for c in classes]
        else:
            # Normalize to lower case for comparison
            reverse_mapping = {v.lower(): k for k, v in cifar10_classes.items()}
            selected_indices = [reverse_mapping[c.lower()] for c in classes]
    except Exception as e:
        print_cust("Error processing classes for CIFAR-10: ", e)
        raise e

    mask = np.isin(y, selected_indices)
    X, y = X[mask], y[mask]

    # Create a mapping similar to load_mnist_digits, ensuring ordering is consistent.
    mapping = {val: idx for idx, val in enumerate(sorted(selected_indices))}

    def to_int(x):
        # If x is a PyTorch tensor, use .item() to extract the value.
        return int(x.item()) if hasattr(x, "item") else int(x)

    # Convert each element in y to a standard integer before mapping.
    y = np.array([mapping[to_int(val)] for val in y])

    # Normalize to [0, 1] and flatten the images.
    X = X.astype("float32") / 255.0
    X = X.reshape(X.shape[0], -1)

    # If needed, subsample the dataset via stratified sampling.
    if n_samples < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)
    return X, y