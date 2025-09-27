from quorus.logging.custom_slog import print_cust
import copy
from quorus.qgan_model_supp.data_proc.pca_rescaler import PCARescaler
from quorus.qgan_model_supp.models.patchquantumgenerator import PatchQuantumGenerator

"""### PatchQuantumGenerator Factory"""

def build_patchquantumgen(data_comps, state_dict, qnode_builder):
  print_cust(f"build_patchquantumgen: called")
  data_comps_list = list(data_comps)
  created_qnode = qnode_builder(data_comps[2])
  data_comps_list[4] = created_qnode
  # TODO: can call build_pca_rescaler here instead, for reusability
  print_cust(f"build_patchquantumgen, type(data_comps_list[-1]): {type(data_comps_list[-1])}")
  print_cust(f"build_patchquantumgen, type(data_comps_list[-1][0]): {type(data_comps_list[-1][0])}")
  data_comps_list[-1] = list(data_comps_list[-1])
  data_comps_list[-1][0] = copy.deepcopy(data_comps_list[-1][0])
  data_comps_list[-1] = PCARescaler(*data_comps_list[-1])
  built_patchquantumgen = PatchQuantumGenerator(*data_comps_list)
  built_patchquantumgen.load_state_dict(state_dict)

  return built_patchquantumgen