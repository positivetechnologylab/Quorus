from pennylane import numpy as np
import pennylane as qml

def flatten_params(params):
    """
    Flatten a list of arrays into a single 1D array.

    Parameters:
      params: a list of numpy arrays

    Returns:
        flat_params: a 1D numpy array containing all parameters
        shapes: a list of shapes (one per original parameter)
    """
    flat_list = []
    shapes = []
    for p in params:
        p = np.array(p)  # ensure it's a Pennylane array
        shapes.append(p.shape)
        flat_list.append(p.flatten())
    flat_params = np.concatenate(flat_list)
    return flat_params, shapes

def unflatten_params(flat_params, shapes):
    """
    Reshape a flat parameter array back into a list of arrays with the given shapes.

    Parameters:
      flat_params: a 1D numpy array containing all parameters
      shapes: a list of shapes (one per original parameter)

    Returns:
      params: a list of numpy arrays
    """
    params_new = []
    index = 0
    for shape in shapes:
        size = int(np.prod(shape))
        param = flat_params[index:index+size].reshape(shape)
        params_new.append(param)
        index += size
    return params_new

def flatten_grad(grad, shapes):
    """
    Flatten a list of gradient arrays into a single 1D array.

    Parameters:
      grad: a list of numpy arrays
      shapes: a list of shapes of the parameters (that the gradients are assumed to follow)

    Returns:
        flat_grad: a 1D numpy array containing all gradients

    The gradients must be provided in the same order as the parameters.
    """
    flat_list = []
    for g, shape in zip(grad, shapes):
        g = np.array(g)
        flat_list.append(g.flatten())
    flat_grad = np.concatenate(flat_list)
    return flat_grad

def flatten_params_recursive(params):
    """

    Parameters:
       params: a list of numpy arrays

    Recursively flattens a nested structure of parameters and returns a 1D numpy array and a list of shapes.

    Returns:
      flat_params: a 1D numpy array containing all parameters
      shapes: a list of shapes (one per original parameter)
    """
    flat_list = []
    shapes = []

    def recurse(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                recurse(sub)
        else:
            arr = np.array(item)
            shapes.append(arr.shape)
            flat_list.append(arr.flatten())

    recurse(params)
    flat_params = np.concatenate(flat_list)
    return flat_params, shapes

def unflatten_params_recursive(flat_params, shapes, structure):
    """
    Rebuilds the nested parameter structure from the flat_params using the recorded shapes and the original structure.

    Parameters:
      flat_params: a 1D numpy array containing all parameters
      shapes: a list of shapes (one per original parameter)
      structure: the original structure of parameters
    """
    flat_elems = []
    pointer = 0
    for shape in shapes:
        size = np.prod(shape)
        flat_elems.append(flat_params[pointer:pointer+size].reshape(shape))
        pointer += size

    it = iter(flat_elems)

    def rebuild(struct):
        if isinstance(struct, (list, tuple)):
            return type(struct)(rebuild(sub) for sub in struct)
        else:
            return next(it)

    return rebuild(structure)

def flatten_grad_recursive(grad):
    """
    grad: a list of numpy arrays representing the gradient

    Recursively flattens a nested structure of gradients into a single 1D numpy array.

    Returns:
      flat_grad: a 1D numpy array containing all gradients
    """
    flat_list = []

    def recurse(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                recurse(sub)
        else:
            arr = np.array(item)
            flat_list.append(arr.flatten())

    recurse(grad)
    flat_grad = np.concatenate(flat_list)
    return flat_grad

"""## Gradient Norm Computation Func"""

def grad_l2_norm(g):
    """
    Compute the global L2 (Euclidean) norm of a nested gradient structure.

    Parameters
    ----------
    g : array‑like | list | tuple
        Gradient tree whose leaves are array‑like tensors.

    Returns
    -------
    float or tensor
        Scalar ‖g‖₂.  Type matches the backend of the leaves
        (NumPy float for vanilla NumPy, 0‑D torch.Tensor for PyTorch, etc.).
    """
    if isinstance(g, (list, tuple)):
        # Recursively accumulate the *squared* norms of the children
        sq_norms = [qml.math.square(grad_l2_norm(subg)) for subg in g]
        return qml.math.sqrt(qml.math.sum(qml.math.stack(sq_norms)))
    else:
        # Leaf: plain array / tensor
        return qml.math.linalg.norm(g)

"""## Tree to list helper"""

def tree_to_list(tree):
    """Return (flat_leaves, rebuild_fn) for an arbitrary nested tree."""
    leaves = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            return [walk(n) for n in node]          # mirror structure
        else:
            leaves.append(node)
            return len(leaves) - 1                  # index placeholder

    structure = walk(tree)

    def rebuild_tree(new_leaves):
        it = iter(new_leaves)
        def r(node):
            if isinstance(node, list):
                return [r(n) for n in node]
            elif isinstance(node, tuple):
                return tuple(r(n) for n in node)
            else:                   # placeholder -> grab next leaf
                return next(it)
        return r(structure)

    return leaves, rebuild_tree