import torch.nn as nn

from quorus.logging.custom_slog import print_cust

"""### Definition of VariationalQuantumClassifier"""

import copy

class VariationalQuantumClassifier(nn.Module):
  """Class for the Variational Quantum Classifier"""
  def __init__(self, qnode, device="cpu"):
    super().__init__()
    # Because these will be passed in.
    self.q_params = None
    self.all_params = None
    self.qnode = qnode
    self.device = device

  def __str__(self):
    return f"""
    VariationalQuantumClassifier,
    self.q_params: {self.q_params}
    self.all_params: {self.all_params}
    self.qnode: {self.qnode}
    self.device: {self.device}
    """

  __repr__ = __str__

  def initialize_existing_parameters(self, existing_params):
    """
    Initializes parameters for the classifier model

    Parameters:
      self (VariationalQuantumClassifier): this object
      existing_params (tuple): A tuple of parameters for this model

    Returns:
      None (sets the parameters according to the provided existing_params)
    """
    print_cust(f"VariationalQuantumClassifier, initialize_existing_parameters, existing_params: {existing_params}")
    self.all_params = copy.deepcopy(list(existing_params))
    param_list = []
    for param_tens in existing_params[5]:
      # TODO, layers: assert that param_tens is a PyTorch tensor?
      param_nnparam = nn.Parameter(param_tens, requires_grad=True)
      param_list.append(param_nnparam)
    self.q_params = nn.ParameterList(param_list)
    self.all_params[5] = self.q_params
    self.all_params = tuple(self.all_params)

  def forward(self, x):
    """
    Function that runs the specified input(s) through the model.

    Parameters:
        x (torch.Tensor): the inputs fed into the model

    Returns:
        the predicted probabilities/outputs from the model.
    """
    # TODO, layers: change this to batched later; trying to make minimal changes.
    # list_tens_probs = []
    # for elem in x:
    #   tens_probs = self.qnode(elem, self.all_params).float()
    #   print_cust(f"VariationalQuantumClassifier, tens_probs: {tens_probs}")
    #   print_cust(f"VariationalQuantumClassifier, tens_probs.shape: {tens_probs.shape}")
    #   print_cust(f"VariationalQuantumClassifier, type(tens_probs): {type(tens_probs)}")
    #   print_cust(f"VariationalQuantumClassifier, tens_probs.dtype: {tens_probs.dtype}")
    #   list_tens_probs.append(tens_probs)
    # stacked_tens_probs = torch.stack(list_tens_probs)
    tens_probs = self.qnode(x, self.all_params).float()
    print_cust(f"VariationalQuantumClassifier, tens_probs.shape: {tens_probs.shape}")
    stacked_tens_probs = tens_probs
    print_cust(f"VariationalQuantumClassifier, stacked_tens_probs: {stacked_tens_probs}")
    print_cust(f"VariationalQuantumClassifier, stacked_tens_probs.shape: {stacked_tens_probs.shape}")
    print_cust(f"VariationalQuantumClassifier, type(stacked_tens_probs): {type(stacked_tens_probs)}")
    print_cust(f"VariationalQuantumClassifier, stacked_tens_probs.dtype: {stacked_tens_probs.dtype}")
    return stacked_tens_probs