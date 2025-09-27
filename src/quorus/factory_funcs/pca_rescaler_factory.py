from quorus.logging.custom_slog import print_cust
from quorus.qgan_model_supp.data_proc.pca_rescaler import PCARescaler

"""## Factory Functions for Objects/Models

### PCARescaler Factory
"""

def build_pca_rescaler(data_comps):
  print_cust(f"build_pca_rescaler, called")
  return PCARescaler(*data_comps)