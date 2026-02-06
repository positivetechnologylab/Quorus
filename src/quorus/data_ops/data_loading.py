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

# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split

def load_cifar10(classes, n_samples=2000):
    """
    Load the CIFAR-10 dataset (without TensorFlow), filter by the specified classes,
    and return the filtered data and labels.

    Parameters
    ----------
    classes : list
        List of classes to keep, either as digit strings like ["3", "5"]
        or names like ["cat", "dog"] (case-insensitive).
    n_samples : int, default=2000
        Number of examples to return (stratified). If >= available, returns all.

    Returns
    -------
    X : np.ndarray, shape (N, 3072), dtype float32
        Image data flattened and normalized to [0, 1].
    y : np.ndarray, shape (N,), dtype int
        Labels remapped to 0..K-1 in sorted order of selected classes.
    """
    # CIFAR-10 standard class names mapping
    _CIFAR10_ID_TO_NAME = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    # --- 1) Load CIFAR-10 from OpenML (no TensorFlow) ---
    # OpenML dataset name for CIFAR-10; returns 32x32x3 images flattened to 3072 features
    cifar = fetch_openml("CIFAR_10", version=1, as_frame=False)
    X = cifar.data          # shape (60000, 3072)
    y = cifar.target        # string labels "0".."9" or class names, depending on version

    # Coerce y to integer class IDs 0..9
    try:
        y_int = y.astype(int)
    except ValueError:
        # If labels are names, map them via CIFAR-10 mapping
        name_to_id = {name.lower(): idx for idx, name in _CIFAR10_ID_TO_NAME.items()}
        y_int = np.array([name_to_id[str(lbl).lower()] for lbl in y], dtype=int)

    # --- 2) Interpret `classes` argument (ids or names) ---
    try:
        if classes and isinstance(classes[0], str) and classes[0].isdigit():
            # e.g. ["3", "5"]
            selected_ids = [int(c) for c in classes]
        else:
            # e.g. ["cat", "dog"]; normalize to lower case
            name_to_id = {name.lower(): idx for idx, name in _CIFAR10_ID_TO_NAME.items()}
            selected_ids = [name_to_id[str(c).lower()] for c in classes]
    except Exception as e:
        print_cust("Error processing CIFAR-10 classes:", e)
        raise

    selected_ids = sorted(set(selected_ids))

    # --- 3) Filter to selected classes ---
    mask = np.isin(y_int, selected_ids)
    X = X[mask]
    y_int = y_int[mask]

    # --- 4) Remap labels to 0..K-1 in sorted order of selected classes ---
    mapping = {orig_id: new_id for new_id, orig_id in enumerate(selected_ids)}
    y = np.array([mapping[int(lbl)] for lbl in y_int], dtype=int)

    # --- 5) Normalize to [0, 1] and flatten (OpenML already gives 3072-d vectors) ---
    X = X.astype("float32") / 255.0
    # If needed, ensure shape (N, 3072)
    X = X.reshape(X.shape[0], -1)

    # --- 6) Optional stratified subsample ---
    if n_samples < len(y):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_samples, stratify=y, random_state=42
        )

    return X, y