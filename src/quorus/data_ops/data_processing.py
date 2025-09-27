from quorus.logging.custom_slog import print_cust

from pennylane import numpy as np

from sklearn.decomposition import PCA

"""## Random Sketching Class"""

# This class adheres to the "fit_transform(X)" and "transform(X)" interface, for compatibility with Numpy type objects.
# This class implements random sketching, given an input sketch matrix.
# TODO: generate the sketch matrix in the class, for better separation of concerns.
class SketchTransformer:
    def __init__(self, sketch_mat: np.ndarray):
        """
        Initialize with a fixed sketch matrix.

        Parameters
        ----------
        sketch_mat : ndarray of shape (d, k)
            The random sketching matrix to be used for all transforms.
        """
        # store the sketch matrix
        self.sketch_mat = sketch_mat

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the sketch to new data X by matrix multiplication.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            The data to be sketched.

        Returns
        -------
        X_sketch : ndarray of shape (n_samples, k)
            The sketched data, X @ sketch_mat.
        """
        # check dimensions
        n_samples, d = X.shape
        d2, k = self.sketch_mat.shape
        if d != d2:
            raise ValueError(f"Input data has dimension {d}, but sketch_mat expects {d2}.")
        # perform the matrix multiplication
        return X @ self.sketch_mat

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the sketch to new data X by matrix multiplication.

        Parameters
        ----------
        X : ndarray of shape (n_samples, d)
            The data to be sketched.

        Returns
        -------
        X_sketch : ndarray of shape (n_samples, k)
            The sketched data, X @ sketch_mat.
        """
        # check dimensions
        n_samples, d = X.shape
        d2, k = self.sketch_mat.shape
        if d != d2:
            raise ValueError(f"Input data has dimension {d}, but sketch_mat expects {d2}.")
        # perform the matrix multiplication
        return X @ self.sketch_mat

def angle_encode_data(X, n_components,
                      y=None,
                      do_pca=False, ret_pca=False,
                      do_lda=False, ret_lda=False,
                      sketch_mat=None, custom_debug=False,
                      ret_orig_data=False):
    """
    Reduce X via PCA or LDA (in this case, instead of LDA, we have random sketching) to n_components, then scale each component to [0, π].

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_components : int
      Number of components to keep.
    y : array-like, shape (n_samples,), optional
      Class labels; required if do_lda=True.
    do_pca : bool, default=False
      If True, perform PCA.
    ret_pca : bool, default=False
      If True and do_pca=True, return (X_angle, pca, X_pca).
    do_lda : bool, default=False
      If True, perform LDA.
    ret_lda : bool, default=False
      If True and do_lda=True, return (X_angle, lda, X_lda).
    sketch_mat : np.ndarray, default=None
      If do_lda=True, then use the provided sketch_mat to perform random sketching.

    Returns
    -------
    X_angle : ndarray, shape (n_samples, n_components)
      Angle-encoded data in [0, π].
    [pca, X_pca] or [lda, X_lda] : only if ret_pca or ret_lda is True
      The fitted model and the raw projected data.
    """
    # 1) reduction
    print_cust(f"angle_encode_data, X.shape: {X.shape}")
    if do_pca and do_lda:
        raise ValueError("Choose exactly one of do_pca or do_lda")
    if do_pca:
        pca = PCA(n_components=n_components, random_state=42)
        X_proj = pca.fit_transform(X)
        print_cust(f"angle_encode_data, pca.explained_variance_ratio_: {pca.explained_variance_ratio_}")
        print_cust(f"angle_encode_data, pca.explained_variance_: {pca.explained_variance_}")
        model, raw = pca, X_proj

    elif do_lda:
        # if y is None:
        #     raise ValueError("y labels are required when do_lda=True")
        # lda = LinearDiscriminantAnalysis(n_components=n_components)
        # X_proj = lda.fit_transform(X, y)
        if sketch_mat is None:
          raise ValueError("sketch_mat is required when do_lda=True")
        sketch_transformer = SketchTransformer(sketch_mat)
        X_proj = sketch_transformer.fit_transform(X)
        model, raw = sketch_transformer, X_proj

        # sklearn's LDA doesn't expose explained_variance_ratio_ by default
        # model, raw = lda, X_proj

    else:
        X_proj = X
        print_cust("angle_encode_data, no reduction applied")
        model, raw = None, X_proj

    # 2) scale each component to [0, π]
    X_angle = np.zeros_like(X_proj)
    for i in range(X_proj.shape[1]):
        comp = X_proj[:, i]
        lo, hi = comp.min(), comp.max()
        X_angle[:, i] = ( (comp - lo) / (hi - lo + 1e-8) ) * np.pi

    # 3) return
    if do_pca and ret_pca:
        return X_angle, model, raw
    if do_lda and ret_lda:
        return X_angle, model, raw
    return X_angle