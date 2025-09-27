from quorus.logging.custom_slog import print_cust
import copy
from quorus.qnn_classifier_model_supp.variational_quantum_classifier import VariationalQuantumClassifier

"""### VariationalQuantumClassifier Factory"""

# this has a slightly different signature; doesn't use state_dict explicitly.
def build_variationalquantumclassifier(data_comps, existing_params, qnode_builder):
  """
  Factory function for building a variational quantum classifier.

  Parameters:
    data_comps (dict): Specifies the configurations of the qnode and classifier.
    existing_params (tuple): Specifies the parameters for the classifier model.
    qnode_builder (func): A function that retruns a Callable with (input_angles, params).

  Returns:
    A VariationalQuantumClassifier object that can be used for classification.
  """
  # data_comps should be [n_data, conv_layers, expansion_data, n_classes, pennylane_interface, device]
  print_cust(f"build_variationalquantumclassifier: called")
  model_device = data_comps["device"]
  # data_comps_qnode = data_comps[:-1]
  data_comps_qnode = copy.deepcopy(data_comps)
  del data_comps_qnode["device"]
  print_cust(f"build_variationalquantumclassifier, data_comps_qnode: {data_comps_qnode}")
  created_qnode = qnode_builder(**data_comps_qnode)
  built_varquantumclassifier = VariationalQuantumClassifier(created_qnode, model_device)
  built_varquantumclassifier.initialize_existing_parameters(existing_params)

  return built_varquantumclassifier
