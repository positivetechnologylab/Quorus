from quorus.logging.custom_slog import print_cust
from quorus.qgan_model_supp.models.pca_discriminator import PCADiscriminator

"""### PCADiscriminator Factory"""

def build_pca_discriminator(data_comps, state_dict):
  print_cust(f"build_pca_discriminator: called")
  built_pca_disc = PCADiscriminator(*data_comps)
  built_pca_disc.load_state_dict(state_dict)
  return built_pca_disc