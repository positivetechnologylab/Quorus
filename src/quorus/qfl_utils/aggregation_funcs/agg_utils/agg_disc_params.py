from quorus.qfl_utils.aggregation_funcs.agg_utils.federated_averaging_disc import fedavg_disc
from quorus.logging.custom_slog import print_cust

"""##### Main Disc Aggregation Function"""

def aggregate_discriminator_params(client_params_dict, clients_data_dict):
  discriminator_models = []
  disc_weights = []

  for client_type, clients_params in client_params_dict.items():
    for params, data in zip(clients_params, clients_data_dict[client_type]):
      # assumed that params[5][1] is indeed the discriminator.
      # no instanceof checking
      discriminator_models.append(params[5][1])
      disc_weights.append(len(data[0][0]))

  disc_model_agg = fedavg_disc(discriminator_models, disc_weights)

  print_cust(f"aggregate_discriminator_params, disc_model_agg: {disc_model_agg}")

  return disc_model_agg